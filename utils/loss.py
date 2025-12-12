"""
损失函数定义模块 (针对 One-hot 序列输入)
========================================
此模块包含用于训练原始 SVAE 模型的损失函数，包括：
1. Gaussian KL Divergence: 计算两个高斯分布之间的差异。
2. Categorical KL Divergence: 计算两个分类分布之间的差异。
3. SVAE Loss: 综合损失函数 (重构误差 + KL 散度 + 回归误差)。
"""

import numpy as np
import tensorflow as tf
from tensorflow import math as tfm
from tensorflow.keras import backend as K

# 定义交叉熵损失 (用于重构误差) 和 MSE (用于回归误差)
CEL_logits = tf.nn.softmax_cross_entropy_with_logits
mse = tf.keras.losses.MeanSquaredError()

# 标准氨基酸列表
AAs = np.array(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])


# -- KLD_Gaussian -- 
# Description: KL divergence of two (multi-dimensional) Gaussian distributions

def KLD_Gaussian(q_mu, q_sigma, p_mu, p_sigma):
    """
    计算两个多元高斯分布 (q 和 p) 之间的 KL 散度。
    公式: KL(q||p) = 0.5 * [log|Σp|/|Σq| - d + tr(Σp^-1 Σq) + (μp-μq)^T Σp^-1 (μp-μq)]
    
    参数:
        q_mu, q_sigma: 分布 q 的均值和对数方差 (通常是 Encoder 的输出)。
        p_mu, p_sigma: 分布 p 的均值和对数方差 (通常是先验分布，如 N(0, I))。
        
    返回:
        KL 散度值 (Batch Size,)
    """
    # 1/2 [log|Σ2|/|Σ1| −d + tr{Σ2^-1 Σ1} + (μ2−μ1)^T Σ2^-1 (μ2−μ1)]
    KLD = 1 / 2 * (2 * (p_sigma - q_sigma)
                   - 1
                   + tf.math.pow(((tf.math.exp(q_sigma)) / (tf.math.exp(p_sigma) + 1e-6)), 2)
                   + tf.math.pow(((p_mu - q_mu) / (tf.math.exp(p_sigma) + 1e-6)), 2))
    return K.sum(KLD, axis=-1)


# -- KLD_Categorical -- 
# Description: KL divergence of two categorical distributions

def KLD_Categorical(q, p):
    """
    计算两个分类分布之间的 KL 散度。
    公式: sum(q * log(q/p))
    """
    # sum (q log (q/p) )
    KLD = q * tf.math.log((q + 1e-4) / (p + 1e-4))
    return K.sum(KLD, axis=-1)



# -- loss -- 
# Description: computes VAE loss of given vae model on (input, label) pair 
# (X,Y). Optionally takes CV (coefficient of variation) of Y values into account
# dividing regression loss by CV+1 (so points with higher uncertainty incur
# lower loss penalty).


def loss(vae, X, Y, 
           CV=0, 
           alphabet=AAs, 
           loss_weights=[0.2, 1.0, 0.2], 
           kind='default'): 
    """
    SVAE 综合损失函数计算。
    
    参数:
        vae (keras.Model): SVAE 模型实例。
        X (tensor): 输入序列 One-hot 张量 (Batch, Seq_Len * 20)。
        Y (tensor): 真实标签值 (Batch, 1)。
        CV (tensor or 0): 标签的变异系数，用于加权回归损失。CV 越大，不确定性越高，Loss 权重越低。
        alphabet (list): 氨基酸字母表。
        loss_weights (list): 三个权重的列表 [重构权重, KL权重, 回归权重]。
        kind (str): KL 散度的计算方式 ('default' 或 'linear')。
        
    返回:
        tuple: (total_loss, reconstruction_loss, kl_loss, regression_loss)
    """
    
    input_dim = tf.shape(X)[1].numpy()
    
    # 计算序列长度 (mer)
    mer = int(input_dim / len(alphabet))
    
    encoder = vae.get_layer('encoder')
    decoder = vae.get_layer('decoder')
    regressor = vae.get_layer('regressor')
    
    # 1. 前向传播
    z_mean, z_log_var, z = encoder(X)
    reconstruction, y_preds = vae(X)

    # 2. 重构损失 (Reconstruction Loss)
    # 对于 One-hot 序列，逐位置计算 Cross Entropy
    reconstruction_loss = 0
    for i in range(mer):
        # 提取第 i 个位置的 logits 和 labels
        logits = reconstruction[:, i*len(alphabet):(i+1)*len(alphabet)]
        labels = X[:, i*len(alphabet):(i+1)*len(alphabet)]
        reconstruction_loss += CEL_logits(labels, logits)
    
    # 3. KL 散度损失 (KL Loss)
    # 计算 Latent 分布与标准正态分布 N(0, I) 的差异
    if kind == 'default':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
    elif kind == 'linear':
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean - tf.reduce_mean(z_mean)) - tf.exp(z_log_var) + 1)
    
    # 4. 回归损失 (Regression Loss)
    regression_loss = mse(y_true=Y, y_pred=y_preds)
    
    # 5. 总损失加权求和
    # 注意：回归损失除以 (CV+1) 是为了处理实验数据的不确定性
    vae_loss = (loss_weights[0]*tf.cast(reconstruction_loss, tf.float64) + 
                loss_weights[1]*tf.cast(kl_loss, tf.float64) + 
                loss_weights[2]*tf.cast(regression_loss, tf.float64)/(CV+1)
               )
    
    return (vae_loss, reconstruction_loss, kl_loss, regression_loss)


# -- grad -- 
# Description: computes VAE gradient of given vae model on (input, label) pair (X,Y) (w.r.t. defined loss)

def grad(model, X, Y):
    with tf.GradientTape() as tape:
        vae_loss, _, _, _ = loss(model, X, Y)
    return tape.gradient(vae_loss, model.trainable_variables)
