"""
SVAE 训练模块
=============
此模块包含 SVAE 模型的训练逻辑，包括：
1. 训练循环 (Training Loop)
2. 验证循环 (Validation Loop)
3. 损失记录 (Loss Tracking)
4. 早停机制 (Early Stopping)
5. WandB 集成
"""

import os, re
import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from pathlib import Path

from svae.predict import predict
from utils.loss import loss, grad
from tensorflow.keras.utils import Progbar

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None


# -- train -- (incorporates CV)
# trains model, returns (model, train_losses, val_losses, preds_df)


def train(
    model,
    train_batches,
    val_batches,
    train_df,
    optimizer=None,
    model_outdir=None,  # Directory for saving model and logs
    loss_weights=[1.0, 0.5, 0.1],
    patience=10,  # Number of epochs to continue training for after convergence
    min_epochs=50,
    max_epochs=100,
    convergence_threshold=0.005,
    progbar_verbosity=1,  # Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
    log_wandb=False,
    wandb_project=None,
    wandb_config=None,
    wandb_run=None,
    wandb_watch=False,
):
    """
    SVAE 模型训练主函数。
    
    参数:
        model: 已编译的 SVAE 模型。
        train_batches: 训练集 TF Dataset。
        val_batches: 验证集 TF Dataset。
        train_df: 训练集原始 DataFrame (用于预测)。
        optimizer: 优化器 (默认 Adam)。
        model_outdir: 模型保存目录。
        loss_weights: Loss 权重 [重建, KL, 回归]。
        patience: 早停耐心值 (Convergence 后继续训练的轮数)。
        min_epochs: 最小训练轮数。
        
    返回：
        model：训练后的模型。
        preds_df：训练集上的预测结果。
        model_outdir：模型保存路径。
    """
    
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam(1e-3)

    # 获取批次信息
    # 获取关于批次的一些信息 - 从批次中推断这些值虽然效率不高（因为它们本就是传入的参数），
    # 但不用重复输入参数也挺方便的...
    batches_list = list(train_batches.as_numpy_iterator())
    batch_size = batches_list[0][0].shape[0]
    last_batch_size = batches_list[-1][0].shape[0]
    num_batches = len(batches_list)
    num_training_samples = (num_batches - 1) * batch_size + last_batch_size

    Y_colname = [col for col in train_df.columns if "Y--" in col][0]
    assay = Y_colname.split("Y--")[-1].split("-")[0]

    wandb_active = False
    wandb_run_ref = None
    wandb_started_here = False
    if log_wandb:
        if wandb is None:
            raise ImportError("log_wandb=True but wandb is not installed. Run `uv add wandb` first.")
        config_payload = {
            "assay": assay,
            "loss_weights": loss_weights,
            "patience": patience,
            "min_epochs": min_epochs,
            "max_epochs": max_epochs,
            "convergence_threshold": convergence_threshold,
            "batch_size": batch_size,
            "num_training_samples": int(num_training_samples),
            "model_name": getattr(model, "name", "svae"),
        }
        if wandb_config:
            config_payload.update(wandb_config)

        if wandb_run is not None:
            wandb_run_ref = wandb_run
        else:
            init_kwargs = {"project": wandb_project or "aav-capsid-receptor"}
            wandb_run_ref = wandb.init(config=config_payload, **init_kwargs)
            wandb_started_here = True

        if wandb_run_ref is not None:
            wandb_active = True
            if wandb_watch:
                wandb.watch(model, log="all", log_freq=100)
            wandb_run_ref.config.update(config_payload, allow_val_change=True)

    ## -- Metrics -- ##
    # model loss: 包含重构、KL 和回归损失的总损失
    # reconstruction loss: (严格的 VAE 损失) 输入 One-hot 序列与解码输出 One-hot 之间的二元交叉熵
    # kl loss: (统计学) 分布之间的差异 - 本质上是一种正则化形式
    # regression loss: 真实 log2enr 值与回归器预测值之间的均方误差 (MSE)

    train_metrics = [
        "train_model_loss",
        "train_reconstruction_loss",
        "train_kl_loss",
        "train_regression_loss",
    ]
    val_metrics = [
        "val_model_loss",
        "val_reconstruction_loss",
        "val_kl_loss",
        "val_regression_loss",
    ]

    metrics_names = train_metrics + val_metrics

    print("Logging following metrics: {}".format(metrics_names))
    print("\n\n----- Beginning training. -----")

    # Losses logged to file
    # 记录到文件的损失
    train_losses = []
    val_losses = []

    # Keeps track of max difference in loss values between consecutive epochs, across all training losses (model total, reconstruction, kl, regression)
    # 跟踪连续 epoch 之间损失值的最大差异，涵盖所有训练损失（模型总损失、重构、KL、回归）
    convergence_history = []

    # After convergence, model continues to train (stalls) for number of epochs specified by 'patience' parameter
    # 收敛后，模型继续训练（停滞）'patience' 参数指定的轮数
    stall = 0
    converged = False
    epochs_run = 0

    #### ---- TRAINING ---- ####
    while not converged or stall < patience:
        epoch_train_losses = []
        epoch_val_losses = []
        for i in range(len(train_metrics)):
            # Adding a tf.keras.metrics.Mean() object for each tracked loss that serves
            # as a container for loss values per iteration; after each epoch, each loss
            # container's state gets updated to contain the most recent loss values, and
            # the mean loss over the epoch is computed
            # 为每个跟踪的损失添加一个 tf.keras.metrics.Mean() 对象，作为每次迭代损失值的容器；
            # 每个 epoch 后，每个损失容器的状态会更新为包含最新的损失值，并计算该 epoch 的平均损失
            epoch_train_losses.append(tf.keras.metrics.Mean())
            epoch_val_losses.append(tf.keras.metrics.Mean())

        print("\nepoch {}".format(epochs_run + 1))
        if converged:
            print("Converged. Stall: {}/{}".format(stall, patience))

        # Nice printout of training progress
        # 漂亮的训练进度打印
        progBar = Progbar(
            num_training_samples,
            stateful_metrics=metrics_names,
            verbose=progbar_verbosity,
            interval=0.1,
        )

        ### --- BEGIN EPOCH --- ###
        for i, examples in enumerate(train_batches):
            x = examples[0]
            y = examples[1]

            # Data optionally contains CV values, in addition to x and y values
            # 数据除了 x 和 y 值外，可选地包含 CV 值
            if len(examples) > 2:
                cv = examples[2]
            else:
                cv = 0

            # Compute training losses
            # 计算训练损失
            train_loss_values = loss(model, x, y, cv, loss_weights=loss_weights)
            
            # NaN 检测
            if tf.reduce_any(tf.math.is_nan(train_loss_values[0])):
                print("\nWARNING: NaN loss detected during training. Stopping early.")
                converged = True
                break

            # Backprop step
            # 反向传播步骤
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Update training losses for printing and progbar
            # 更新训练损失以供打印和进度条显示
            for l, loss_val in enumerate(train_loss_values):
                epoch_train_losses[l].update_state(loss_val)
            values = zip(train_metrics, train_loss_values)

            # Update progress bar for every training example throughout the epoch
            # 在整个 epoch 中为每个训练样本更新进度条
            progBar.update(i * batch_size, values=values)

        ### --- END EPOCH --- ###
        train_losses.append([loss_avg.result() for loss_avg in epoch_train_losses])

        #### ---- VALIDATION ---- ####
        for i, examples in enumerate(val_batches):
            batch_x = examples[0]
            batch_y = examples[1]

            if len(examples) > 2:
                batch_cv = examples[2]
            else:
                batch_cv = 0

            # Compute validation losses
            # 计算验证损失
            val_loss_values = loss(
                model,
                batch_x,
                batch_y,
                batch_cv,
                loss_weights=loss_weights,
            )
            # Update validation losses for printing and progbar
            # 更新验证损失
            for l, loss_val in enumerate(val_loss_values):
                epoch_val_losses[l].update_state(loss_val)

        val_losses.append([loss_avg.result() for loss_avg in epoch_val_losses])

        # Report both training and validation losses together at epoch end
        # 在 epoch 结束时同时报告训练和验证损失
        all_loss_avgs = [loss_avg.result() for loss_avg in epoch_train_losses] + [
            loss_avg.result() for loss_avg in epoch_val_losses
        ]
        values = zip(metrics_names, all_loss_avgs)

        # Update progress bar with end-of-epoch training and validation losses
        # 使用 epoch 结束时的训练和验证损失更新进度条
        progBar.update(num_training_samples, values=values, finalize=True)

        if wandb_active:
            epoch_index = epochs_run + 1
            log_payload = {metric: float(loss_val) for metric, loss_val in zip(metrics_names, all_loss_avgs)}
            log_payload["epoch"] = epoch_index
            wandb_run_ref.log(log_payload, step=epoch_index)

        epochs_run += 1

        if epochs_run >= max_epochs:
            print("Hit maximum epochs.")
            converged = True
            break

        if epochs_run < min_epochs:
            continue

        # Compute training loss improvement from previous epoch
        # 计算相比上一个 epoch 的训练损失改善
        if not converged:
            improvement = np.array(train_losses[-1]) - np.array(train_losses[-2])

            if np.max(improvement) < convergence_threshold:
                convergence_history.append(1)
            else:
                convergence_history.append(0)

            print("Convergence history: {}".format(convergence_history))

        # Training has converged if the max loss difference between consecutive epochs
        # is less than convergence threshold for at least 3 of the last 5 runs
        # 如果连续 epoch 之间的最大损失差值小于收敛阈值的次数在过去 5 次中至少占 3 次，则认为训练已收敛
        if len(convergence_history) >= 5:
            recent_history = convergence_history[-5:]
            if not converged and np.sum(recent_history) / (len(recent_history)) >= 0.7:
                converged = True
                stall += 1
                continue

        if stall > 0:
            stall += 1

    final_train_model_loss = train_losses[-1][0]
    final_val_model_loss = val_losses[-1][0]
    print(
        "Finished training model. Final overall losses:\ntrain: {:.3f}    val:{:.3f}".format(
            final_train_model_loss, final_val_model_loss
        )
    )
    #### --- END OF TRAINING --- ####

    # Make dataframes of train and val loss histories and combine into in a single loss dataframe
    # 创建训练和验证损失历史的数据框，并合并为单个损失数据框
    train_loss_df = pd.DataFrame(
        train_losses,
        columns=[
            "train_model_loss",
            "train_reconstruction_loss",
            "train_kl_loss",
            "train_regression_loss",
        ],
    )
    val_loss_df = pd.DataFrame(
        val_losses,
        columns=[
            "val_model_loss",
            "val_reconstruction_loss",
            "val_kl_loss",
            "val_regression_loss",
        ],
    )
    loss_df = pd.concat([train_loss_df, val_loss_df], axis=1)

    # Make train regressor predictions on train and val data combined
    # 在合并的训练和验证数据上进行回归预测
    AA = train_df["AA_sequence"].values
    X = train_df[[col for col in train_df.columns if re.match(r"x\d+", col)]].values
    Y = train_df[Y_colname].values
    preds_df = predict(model, X, Y=Y, AA=AA)
    train_df[Y_colname.replace("Y", "y_pred")] = preds_df["y_pred"]
    Z_cols = [col for col in preds_df.columns if re.match(r"z\d+", col)]
    for col in Z_cols:
        train_df[col] = preds_df[col]

    if wandb_active and wandb_run_ref is not None:
        wandb_run_ref.summary["final_train_model_loss"] = float(final_train_model_loss)
        wandb_run_ref.summary["final_val_model_loss"] = float(final_val_model_loss)
        wandb_run_ref.summary["epochs_trained"] = epochs_run
        loss_table = loss_df.copy()
        loss_table.insert(0, "epoch", np.arange(1, len(loss_df) + 1))
        preds_preview = preds_df.head(500).copy()
        wandb_run_ref.log(
            {
                "loss_history": wandb.Table(dataframe=loss_table),
                "train_predictions_preview": wandb.Table(dataframe=preds_preview),
            },
            step=epochs_run,
        )

    # Save everything to file

    if model_outdir is None:
        cwd = Path(os.getcwd())
        top_dir = cwd.parent
        now = datetime.now()
        latent_dim = model.get_layer("encoder").weights[-1].numpy().shape[0]
        model_outdir = Path(
            "{}/trained_models/{}{}{}_{}_{}D_{}epochs_{:.2f}T_{:.2f}V/".format(
                top_dir,
                now.year,
                now.month,
                now.day,
                assay,
                latent_dim,
                epochs_run,
                final_train_model_loss,
                final_val_model_loss,
            )
        )
    model_outdir.mkdir(parents=True, exist_ok=True)

    # Save model
    # 保存模型
    model_outpath = model_outdir / "model"
    tf.keras.models.save_model(model, model_outpath)
    print("Model saved to {}.".format(str(model_outpath)))

    # Save loss df
    # 保存损失数据框
    loss_outpath = model_outdir / "loss_log.csv"
    loss_df.to_csv(loss_outpath, index=False)
    print("Losses saved to {}.".format(str(loss_outpath)))

    # Save train predictions df
    # 保存训练集预测数据框
    preds_outpath = model_outdir / "preds.csv"
    preds_df.to_csv(preds_outpath, index=False)
    print("Train predictions saved to {}.".format(str(preds_outpath)))

    if wandb_active and wandb_run_ref is not None:
        wandb_run_ref.save(str(loss_outpath))
        wandb_run_ref.save(str(preds_outpath))
        if wandb_started_here:
            wandb_run_ref.finish()

    return model, preds_df, model_outdir
