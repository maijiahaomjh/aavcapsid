"""
SVAE训练脚本 - ESM-C Embeddings (高精度优化版)
================================================

优化策略：
1. 增加 Latent Dim (2 -> 32) 以保留更多 Embedding 信息
2. 调整 Loss 权重：大幅提高回归权重，降低 KL 权重
3. 使用 t-SNE 进行高维 Latent Space 的可视化

"""

import os
import sys
import re
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.manifold import TSNE  # 用于高维可视化
from sklearn.decomposition import PCA  # 用于全局 PCA 降维

from svae.SVAE import SVAE
import svae.train
from utils.data_processing import prep_data, make_tf_batches

# 导入自定义的 ESM-C loss 函数
from utils.loss_esmc import loss as esmc_loss
from utils.loss_esmc import grad as esmc_grad

# Monkey Patch
print("应用 Monkey Patch: 使用 ESM-C 专用 Loss 函数")
svae.train.loss = esmc_loss
svae.train.grad = esmc_grad
from svae.train import train

# ============================================================================
# 默认配置参数 (可被 WandB Sweep 覆盖)
# ============================================================================
DEFAULT_CONFIG = {
    "input_dim": 6720,
    "latent_dim": 32,
    "compression_dim": 20,
    "enc_hidden_dims": [2048, 512],
    "dec_hidden_dims": [512, 2048],
    "reg_hidden_dims": [512, 256, 128],
    "loss_weights": [0, 0, 1.0],  # [Rec, KL, Reg]
    "initial_lr": 1e-3,
    "batch_size": 256,
    "patience": 20,
    "min_epochs": 2,
    "max_epochs": 2,
    "l1_reg": 0.0,
    "l2_reg": 0.0
}

# 全局常量保持用于非 WandB 运行时的回退
INPUT_DIM = DEFAULT_CONFIG["input_dim"]
# 训练集比例调整：测试集 0.1 (即 90% 用于训练+验证)
TEST_SIZE = 0.1
VAL_SIZE = 0.1 # 验证集占剩余的 10% (即总体的 9%)

# WandB 配置
USE_WANDB = False
WANDB_PROJECT = "aav-pulldown-svae-esmc"
WANDB_ENTITY = "767786473-jilin-university"

print("\n" + "=" * 80)
print("SVAE 高精度优化版配置 (支持 WandB Sweep)")
print("=" * 80)

if USE_WANDB:
    import wandb

# ============================================================================
# 数据加载 (与之前相同)
# ============================================================================

def load_esmc_embeddings():
    # 使用 Flattened 7AA 版本
    # CSV 使用原始版本 (包含标签)，NPY 使用处理后的版本
    csv_path = Path('/home/maijiahao/fit4function/data/aavpulldown_4-10_dedup_esmc300_embeddings.csv')
    npy_path = Path('/home/maijiahao/fit4function/data/aavpulldown_4-10_dedup_esmc300_embeddings_flattened_7aa.npy')
    
    print(f"\n加载数据 (Flattened 7AA)...")
    if npy_path.exists():
        embeddings = np.load(npy_path)
        df = pd.read_csv(csv_path)
    else:
        # 如果 npy 不存在但 csv 存在 (应该不会发生，除非手动删了)
        df = pd.read_csv(csv_path)
        emb_cols = [c for c in df.columns if c.startswith('emb_')]
        embeddings = df[emb_cols].values
    
    # 验证维度
    if embeddings.shape[1] != INPUT_DIM:
        print(f"警告: 数据维度 ({embeddings.shape[1]}) 与 INPUT_DIM ({INPUT_DIM}) 不匹配!")
        # 尝试自动修正 INPUT_DIM? 不，最好手动确认。
        # 但如果是 6720，那就对了。

    rename_dict = {}
    if 'AAs' in df.columns: rename_dict['AAs'] = 'AA_sequence'
    elif 'sequence' in df.columns: rename_dict['sequence'] = 'AA_sequence'
    if 'NC Log2 富集值' in df.columns: rename_dict['NC Log2 富集值'] = 'NC_Log2'
    if 'TFRC Log2 富集值' in df.columns: rename_dict['TFRC Log2 富集值'] = 'TFRC_Log2'
    if rename_dict: df = df.rename(columns=rename_dict)
    
    return df, embeddings

# ============================================================================
# 改进的可视化函数 (支持 t-SNE)
# ============================================================================

def create_visualization_optimized(preds_df, target_name, output_dir):
    """创建回归分析可视化图表 (包含 t-SNE)"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    has_y_true = 'y_true' in preds_df
    y_true = preds_df['y_true'].values if has_y_true else None
    y_pred = preds_df['y_pred'].values
    
    if has_y_true:
        pearson_r, _ = pearsonr(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        pearson_r, r2, rmse = np.nan, np.nan, np.nan
    
    # 优先使用 Compression 特征 (c0, c1, ...)，否则回退到 Latent z
    comp_cols = [c for c in preds_df.columns if c.startswith('c')]
    z_cols = [c for c in preds_df.columns if c.startswith('z')]
    
    if comp_cols:
        tsne_source = preds_df[comp_cols].values
        tsne_title = f'Compression Space t-SNE ({len(comp_cols)}D)'
    elif z_cols:
        tsne_source = preds_df[z_cols].values
        tsne_title = f'Latent Space t-SNE ({len(z_cols)}D)'
    else:
        tsne_source = None
        tsne_title = 't-SNE'
    
    if tsne_source is not None:
        print(f"  正在进行 t-SNE 降维 ({tsne_source.shape[1]}D -> 2D)...")
        if len(tsne_source) > 5000:
            indices = np.random.choice(len(tsne_source), 5000, replace=False)
        else:
            indices = np.arange(len(tsne_source))
        
        feat_sample = tsne_source[indices]
        y_sample = y_true[indices] if has_y_true else None
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        feat_2d = tsne.fit_transform(feat_sample)
        
        if has_y_true:
            quantiles = np.percentile(y_sample, [33, 66])
            bins = [-np.inf, quantiles[0], quantiles[1], np.inf]
            cls = np.digitize(y_sample, bins) - 1
            class_colors = np.array(['#1b9e77', '#d95f02', '#7570b3'])
        else:
            cls = None
    else:
        feat_2d = None
        y_sample = None
        cls = None
    
    num_plots = 2 if not has_y_true else 4
    fig, axes = plt.subplots(1, num_plots, figsize=(6*num_plots, 5))
    if num_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    if has_y_true:
        ax = axes[plot_idx]
        ax.scatter(y_true, y_pred, alpha=0.3, s=10)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_title(f'{target_name} Pred vs True\nR={pearson_r:.3f}, R2={r2:.3f}')
        ax.set_xlabel('True')
        ax.set_ylabel('Pred')
        plot_idx += 1
        
        ax = axes[plot_idx]
        ax.scatter(y_pred, y_pred - y_true, alpha=0.3, s=10)
        ax.axhline(0, color='r', linestyle='--')
        ax.set_title(f'Residuals (RMSE={rmse:.3f})')
        ax.set_xlabel('Pred')
        ax.set_ylabel('Pred - True')
        plot_idx += 1
    
    if feat_2d is not None:
        ax = axes[plot_idx]
        scatter = ax.scatter(
            feat_2d[:, 0], feat_2d[:, 1],
            c=y_sample if has_y_true else None,
            cmap='viridis' if has_y_true else None,
            alpha=0.6, s=15
        )
        if has_y_true:
            plt.colorbar(scatter, ax=ax, label='True Value')
        ax.set_title(tsne_title)
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        plot_idx += 1
        
        if has_y_true and cls is not None:
            ax = axes[plot_idx]
            for cat in range(3):
                mask = cls == cat
                ax.scatter(
                    feat_2d[mask, 0], feat_2d[mask, 1],
                    c=class_colors[cat], label=f'Bin {cat+1}', alpha=0.6, s=18
                )
            ax.set_title('t-SNE (离散分箱)')
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{target_name}_optimized_analysis.png', dpi=150)
    plt.close()
    
    return r2, pearson_r

# ============================================================================
# 训练流程
# ============================================================================

def train_model(df, embeddings, target_name, target_col):
    print(f"\n训练 {target_name} 模型...")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # WandB Init (moved to start of function to capture config)
    wandb_run = None
    config = DEFAULT_CONFIG.copy()
    
    if USE_WANDB:
        run_name = f"{target_name}_Sweep_{timestamp}"
        # 注意: 如果是在 Sweep Agent 中运行，wandb.init 会自动合并 Sweep 参数到 config
        wandb_run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            config=DEFAULT_CONFIG,
            reinit=True,
            group=f"Sweep_{timestamp}" # Group runs together
        )
        # 更新 config 为 wandb.config (包含了 Sweep 参数)
        # wandb.config 可能是特殊的 Config 对象，转换为 dict 或直接访问
        for k in DEFAULT_CONFIG.keys():
            if k in wandb.config:
                config[k] = wandb.config[k]
        
        # 支持从 Sweep 调整 Loss 权重分量
        if "loss_weight_rec" in wandb.config:
            config["loss_weights"] = [
                wandb.config.get("loss_weight_rec", 0.0),
                wandb.config.get("loss_weight_kl", 0.0),
                wandb.config.get("loss_weight_reg", 1.0)
            ]
                
        print(f"  WandB Config Loaded: {config}")

    # 从 config 提取参数
    input_dim = config["input_dim"]
    latent_dim = config["latent_dim"]
    compression_dim = config["compression_dim"]
    loss_weights = config["loss_weights"]
    initial_lr = config["initial_lr"]
    batch_size = config["batch_size"]
    
    from sklearn.model_selection import train_test_split
    
    X = embeddings.astype(np.float32)
    Y = df[target_col].values.astype(np.float32)
    
    indices = np.arange(len(df))
    X_train_full, X_test, Y_train_full, Y_test, idx_train, idx_test = train_test_split(
        X, Y, indices, test_size=TEST_SIZE, random_state=42
    )
    
    train_df = df.iloc[idx_train].copy()
    test_df = df.iloc[idx_test].copy()
    
    train_df[f"Y--{target_col}"] = Y_train_full
    test_df[f"Y--{target_col}"] = Y_test
    
    # 添加 x 列
    x_cols = [f'x{i}' for i in range(input_dim)]
    train_X_df = pd.DataFrame(X_train_full, columns=x_cols, index=train_df.index)
    train_df = pd.concat([train_df, train_X_df], axis=1)
    
    # 添加 test x 列 (为预测做准备)
    test_X_df = pd.DataFrame(X_test, columns=x_cols, index=test_df.index)
    test_df = pd.concat([test_df, test_X_df], axis=1)
    
    print(f"  特征维度: {input_dim}, 训练样本: {len(train_df)}")
    
    # Batches
    train_batches, val_batches, _ = make_tf_batches(
        train_df[x_cols], 
        pd.Series(Y_train_full),
        val_size=VAL_SIZE,  # 使用全局配置的 VAL_SIZE
        batch_size=batch_size,
        shuffle=True
    )
    
    # Model (Standard SVAE)
    model = SVAE(
        input_dim=input_dim,
        latent_dim=latent_dim,
        enc_hidden_dims=config["enc_hidden_dims"],
        dec_hidden_dims=config["dec_hidden_dims"],
        reg_hidden_dims=config["reg_hidden_dims"],
        compression_dim=compression_dim,
        l1_reg=config.get("l1_reg", 0.0),
        l2_reg=config.get("l2_reg", 0.0)
    )
    
    model_outdir = Path('/home/maijiahao/trained_models') / f"{timestamp}_{target_name}_ESMC_Sweep"
    model_outdir.mkdir(parents=True, exist_ok=True)
    
    # 添加 clipnorm 防止梯度爆炸 (NaN)
    optimizer = tf.keras.optimizers.Adam(initial_lr, clipnorm=1.0)
    
    model, preds_df, model_outdir = train(
        model=model,
        train_batches=train_batches,
        val_batches=val_batches,
        train_df=train_df,
        optimizer=optimizer,
        model_outdir=model_outdir,
        loss_weights=loss_weights,
        patience=config["patience"],
        min_epochs=config["min_epochs"],
        max_epochs=config["max_epochs"],
        convergence_threshold=0.001,
        log_wandb=USE_WANDB,
        wandb_run=wandb_run
    )
    
    # 训练集评估
    print("  评估训练集...")
    create_visualization_optimized(preds_df, target_name, model_outdir / 'train_analysis')
    
    # 上传训练集图片
    if USE_WANDB and wandb_run:
        train_img_path = model_outdir / 'train_analysis' / f'{target_name}_optimized_analysis.png'
        if train_img_path.exists():
            wandb_run.log({"train_analysis_plot": wandb.Image(str(train_img_path))})
    
    # 测试集评估
    print("  评估测试集...")
    from svae.predict import predict
    AA_test = test_df["AA_sequence"].values
    # predict 接受 numpy X，但我们可以直接传 X_test
    test_preds_df = predict(model, X_test, Y=Y_test, AA=AA_test)
    test_preds_df.to_csv(model_outdir / 'test_preds.csv', index=False)
    
    r2_test, pearson_test = create_visualization_optimized(test_preds_df, f"{target_name}_Test", model_outdir / 'test_analysis')
    
    print(f"  {target_name} 测试集 R2: {r2_test:.4f}, Pearson: {pearson_test:.4f}")

    # 上传测试集指标和图片
    if USE_WANDB and wandb_run:
        test_img_path = model_outdir / 'test_analysis' / f'{target_name}_Test_optimized_analysis.png'
        if test_img_path.exists():
            wandb_run.log({
                "test_analysis_plot": wandb.Image(str(test_img_path)),
                "test_r2": r2_test,
                "test_pearson": pearson_test
            })
        wandb_run.finish()

if __name__ == "__main__":
    df, embeddings = load_esmc_embeddings()
    # 仅训练 TFRC 用于 Sweep 调参
    # 如果需要训练 NC，请另行运行或修改此处
    train_model(df, embeddings, 'TFRC', 'TFRC_Log2')
    # train_model(df, embeddings, 'NC', 'NC_Log2')
    print("\n优化训练完成！")
