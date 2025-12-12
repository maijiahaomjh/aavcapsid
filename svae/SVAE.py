"""SVAE model
Author: Hikari Sorensen - Vector Engineering Team (hsorense@broadinstitute.org)
Notes: 
- this model uses TensorFlow's functional API
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Reshape, TimeDistributed, Flatten, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import L1L2

# activation function
ELU = tf.nn.elu


# -- reparam_trick -- 
# see Kingma DP, Salimans T and Welling M. 2015. Variational Dropout and the Local Reparameterization Trick. Advances in Neural Information Processing Systems. https://proceedings.neurips.cc/paper/2015/file/bc7316929fe1545bf0b98d114ee3ecb8-Paper.pdf

def reparam_trick(z_mean, z_log_var):
    """
    Reparameterization Trick (重参数化技巧)。
    
    目的: 使采样操作变得可导，从而允许梯度反向传播。
    原理: z = mu + sigma * epsilon，其中 epsilon ~ N(0, 1)
    
    参数:
        z_mean: Latent 分布的均值。
            
    返回：
        tensor：采样得到的潜变量 z。
    """
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    # 默认情况下，random_normal 的均值为 0，标准差为 1.0
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon


# -------- ENCODER --------
def Encoder(enc_input_dim, enc_output_dim, enc_hidden_dims, compression_dim=None, l1_reg=0.0, l2_reg=0.0):
    """
    构建并返回一个编码器模型实例。
    
    参数：
        enc_input_dim：输入特征维度。
        enc_output_dim：输出潜变量维度。
        enc_hidden_dims：编码器隐藏层的神经元数量列表。
        compression_dim: (Optional) 局部特征压缩维度。如果指定，将先对每个位置的 embedding 进行降维。
        l1_reg: L1 正则化系数。
        l2_reg: L2 正则化系数。
        
    返回：
        keras.Model：构建好的编码器模型。
                     Inputs：[encoder_input]
                     Outputs：[z_mean, z_log_var, z]
    """
    
    latent_dim = enc_output_dim
    regularizer = L1L2(l1=l1_reg, l2=l2_reg)

    enc_inputs = Input(shape=(enc_input_dim,), name='encoder_input')
    
    x = enc_inputs
    
    # 局部特征压缩层 (Optional)
    # 将 [Batch, 6720] -> [Batch, 7, 960] -> [Batch, 7, compression_dim] -> [Batch, 7*compression_dim]
    if compression_dim:
        # 假设序列长度为 7 (ESM-C Flattened 7AA)
        seq_len = 7
        token_dim = enc_input_dim // seq_len
        
        x = Reshape((seq_len, token_dim))(x)
        x = TimeDistributed(Dense(compression_dim, activation='elu', name='enc_compression', kernel_regularizer=regularizer))(x)
        x = Flatten()(x)
    
    # 捕获压缩后（或原始）的特征，用于回归器的输入
    x_feat = x
    
    enc_hidden = [Dense(dim, activation='linear', kernel_regularizer=regularizer,
            name='enc_hidden_{}'.format(i)) for i, dim in enumerate(enc_hidden_dims)]
    
    z_mean_layer = Dense(latent_dim, activation='linear', name='z_mean', kernel_regularizer=regularizer)
    z_log_var_layer = Dense(latent_dim, activation='linear', name='z_log_var', kernel_regularizer=regularizer)
    
    # 构建编码器模型
    for layer in enc_hidden:
        x = ELU(layer(x))

    z_mean = z_mean_layer(x)
    z_log_var = z_log_var_layer(x)
    
    # 使用重参数化技巧将采样操作推迟到输入层之外
    # 注意：在 TensorFlow 后端中，"output_shape" 参数其实不是必须的
    z = reparam_trick(z_mean, z_log_var)
    
    
    # 实例化编码器模型
    # Output[3] 为 x_feat (压缩后的特征 或 原始特征)
    encoder = Model(enc_inputs, [z_mean, z_log_var, z, x_feat], name='encoder')
    
    return encoder, enc_inputs


# --------- DECODER -------
def Decoder(dec_input_dim, dec_output_dim, dec_hidden_dims, l1_reg=0.0, l2_reg=0.0):
    """
    构建并返回一个解码器模型实例。
    
    参数：
        dec_input_dim：输入潜变量维度。
        dec_output_dim：输出重构序列维度。
        dec_hidden_dims：解码器隐藏层的神经元数量列表。
        l1_reg: L1 正则化系数。
        l2_reg: L2 正则化系数。
        
    返回：
        keras.Model：构建好的解码器模型。
                     Inputs：[decoder_input]
                     Outputs：[decoder_output]
    """
    
    latent_dim = dec_input_dim
    regularizer = L1L2(l1=l1_reg, l2=l2_reg)
    
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    dec_hidden = [Dense(dim, activation='linear', kernel_regularizer=regularizer,
            name='dec_hidden_{}'.format(i)) for i, dim in enumerate(dec_hidden_dims)]
    
    dec_output_layer = Dense(dec_output_dim, activation='linear', kernel_regularizer=regularizer)
    
    # 解码器计算
    x = latent_inputs
    for layer in dec_hidden:
        x = ELU(layer(x))

    dec_outputs = dec_output_layer(x)

    # 实例化解码器模型
    decoder = Model(latent_inputs, dec_outputs, name='decoder')
    
    return decoder


# -------- REGRESSOR -----------
def Regressor(latent_dim, enc_input_dim, reg_hidden_dims, l1_reg=0.0, l2_reg=0.0):
    """
    构建并返回一个回归器模型实例。
    
    参数：
        latent_dim：输入潜变量维度。
        enc_input_dim：输入原始序列维度。
        reg_hidden_dims：回归器隐藏层的神经元数量列表。
        l1_reg: L1 正则化系数。
        l2_reg: L2 正则化系数。
        
    返回：
        keras.Model：构建好的回归器模型。
                     Inputs：[regressor_input]
                     Outputs：[regressor_output]
    """
    
    regularizer = L1L2(l1=l1_reg, l2=l2_reg)
    reg_latent_inputs = Input(shape=(latent_dim+enc_input_dim,), name='regressor_input')
    reg_layerz = Dense(reg_hidden_dims[0], activation='linear', name='reg_z', kernel_regularizer=regularizer)
    reg_layerx = Dense(reg_hidden_dims[0], activation='linear', name='reg_x', kernel_regularizer=regularizer)
    reg_hidden = [Dense(dim, activation='linear', kernel_regularizer=regularizer,
            name='reg_hidden_{}'.format(i)) for i, dim in enumerate(reg_hidden_dims[1:])]
    reg_output_layer = Dense(1, activation='linear', name='reg_output', kernel_regularizer=regularizer)
    
    
    # 回归器同时接收原始 One-hot 编码和变换后的潜变量表示作为输入
    z_input = reg_latent_inputs[:, :latent_dim]
    x_input = reg_latent_inputs[:, latent_dim:]
    
    # 回归器计算
    if latent_dim > 0:
        reg_z = ELU(reg_layerz(z_input))
        reg_x = ELU(reg_layerx(x_input))
        x = Concatenate()([reg_z, reg_x])
    else:
        # 如果 latent_dim 为 0，仅使用 x_input
        reg_x = ELU(reg_layerx(x_input))
        x = reg_x
        
    for layer in reg_hidden:
        x = ELU(layer(x))
    reg_outputs = reg_output_layer(x)
    
    # 实例化回归器模型
    regressor = Model(reg_latent_inputs, reg_outputs, name='regressor')
    
    return regressor



# -- SVAE -- 

def SVAE(input_dim=140, 
        latent_dim=2, 
        enc_hidden_dims=[100,40], 
        dec_hidden_dims=[40,100],
        reg_hidden_dims=[100,10],
        compression_dim=None,
        l1_reg=0.0,
        l2_reg=0.0,
        name='svae'
       ):
    
    encoder, enc_inputs = Encoder(input_dim, latent_dim, enc_hidden_dims, compression_dim=compression_dim, l1_reg=l1_reg, l2_reg=l2_reg)
    decoder = Decoder(latent_dim, input_dim, dec_hidden_dims, l1_reg=l1_reg, l2_reg=l2_reg)
    if reg_hidden_dims and reg_hidden_dims != None:
        
        # 计算回归器的输入特征维度 (仅使用 x部分，忽略 latent)
        if compression_dim:
            # 如果启用了压缩，则维度为 7 * compression_dim
            reg_input_dim = 7 * compression_dim
        else:
            # 否则为原始输入维度
            reg_input_dim = input_dim
            
        # 注意：这里我们不再传入 latent_dim 给 Regressor，或者将其设为 0
        # 为了复用 Regressor 类 (它期望 latent_dim 参数)，我们可以传 0
        regressor = Regressor(0, reg_input_dim, reg_hidden_dims, l1_reg=l1_reg, l2_reg=l2_reg)
        
        # 获取 Encoder 的所有输出
        enc_outs = encoder(enc_inputs)
        # enc_outs[3] = x_feat (压缩特征 或 原始特征)
        
        # 配置完整的 SVAE 输出（含回归器）
        # Regressor 输入: 仅 x_feat (忽略 Latent Z)
        reg_input = enc_outs[3]
        
        model_outputs = [decoder(enc_outs[2]), regressor(reg_input)]
    else:
        # 配置完整的 VAE 输出（不含回归器）
        model_outputs = [decoder(encoder(enc_inputs)[2])]
    
    # 实例化完整的 SVAE 模型
    model = Model(enc_inputs, model_outputs, name=name)

    return model
