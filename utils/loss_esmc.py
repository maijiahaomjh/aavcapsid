"""
Loss function for ESM-C Embeddings (Continuous Values)
======================================================
此模块是 utils/loss.py 的修改版，专门用于处理连续的 Embedding 输入。

主要区别：
1. 重构损失 (Reconstruction Loss): 使用 MSE (均方误差) 而不是 Cross Entropy。
   - 因为 Embedding 是连续数值，不是离散分类。
2. 输入形状处理: 不再拆分 alphabet 和 seq_len，直接对整个向量计算 Loss。

作者: Mai Jiahao
"""

import tensorflow as tf
from tensorflow.keras import backend as K

mse = tf.keras.losses.MeanSquaredError()

def loss(vae, X, Y, 
           CV=0, 
           loss_weights=[1.0, 0.5, 0.1], 
           kind='default'): 
    """
    针对 Embedding 输入的 SVAE 损失函数。
    
    参数:
        vae (keras.Model): SVAE 模型。
        X (tensor): 输入 Embedding 张量 (Batch, Embedding_Dim)。
        Y (tensor): 真实标签。
        CV (tensor or 0): 变异系数。
        loss_weights (list): [重构权重, KL权重, 回归权重]。
        kind (str): KL 计算方式。
        
    返回:
        tuple: (total_loss, reconstruction_loss, kl_loss, regression_loss)
    """
    
    encoder = vae.get_layer('encoder')
    decoder = vae.get_layer('decoder')
    # Regressor is optional
    try:
        regressor = vae.get_layer('regressor')
        has_regressor = True
    except:
        has_regressor = False

    # 1. 前向传播
    # 获取 Latent 变量
    # Encoder 可能返回 3 个值 (z_mean, z_log_var, z) 或 4 个值 (..., x_feat)
    # 我们只需要前三个用于计算 KL Loss
    enc_outs = encoder(X)
    z_mean = enc_outs[0]
    z_log_var = enc_outs[1]
    z = enc_outs[2]
    
    # 获取输出
    # 如果有 Regressor，vae(X) 返回 [reconstruction, y_preds]
    # 否则只返回 reconstruction
    if has_regressor:
        reconstruction, y_preds = vae(X)
    else:
        reconstruction = vae(X)
        y_preds = None

    # 2. 重构损失 (Reconstruction Loss)
    # [关键修改] 使用 MSE 计算 Embedding 之间的欧氏距离
    # X 和 reconstruction 形状都是 (batch, input_dim)
    reconstruction_loss = mse(X, reconstruction)
    
    # 3. KL 散度损失 (KL Loss)
    # 计算 Latent 分布与标准正态分布的差异
    if kind == 'default':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    elif kind == 'linear':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean - tf.reduce_mean(z_mean)) - tf.exp(z_log_var) + 1)
    
    # 4. 回归损失 (Regression Loss)
    if has_regressor:
        regression_loss = mse(y_true=Y, y_pred=y_preds)
        
        # 5. 总损失求和
        vae_loss = (loss_weights[0]*tf.cast(reconstruction_loss, tf.float64) + 
                    loss_weights[1]*tf.cast(kl_loss, tf.float64) + 
                    loss_weights[2]*tf.cast(regression_loss, tf.float64)/(CV+1)
                   )
    else:
        regression_loss = 0.0
        vae_loss = (loss_weights[0]*tf.cast(reconstruction_loss, tf.float64) + 
                    loss_weights[1]*tf.cast(kl_loss, tf.float64)
                   )
    
    return (vae_loss, reconstruction_loss, kl_loss, regression_loss)

def grad(model, X, Y):
    """
    计算梯度。
    """
    with tf.GradientTape() as tape:
        vae_loss, _, _, _ = loss(model, X, Y)
    return tape.gradient(vae_loss, model.trainable_variables)
