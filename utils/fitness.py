"""
活性预测工具模块
================
此模块提供了一个便捷函数，用于加载预训练的 Keras 模型并对给定的氨基酸序列进行活性 (Fitness) 预测。
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# 修正导入路径
try:
    from .data_processing import *
except ImportError:
    # 尝试从 utils 直接导入 (如果作为脚本运行)
    from utils.data_processing import *

def predict_fitness(AA_sequence, 
                    model=None,
                    model_file='/mnt/shared/notebooks/albert/FitPred/Final_model_24K_v1.h5'
                   ):
    """
    预测给定氨基酸序列的活性值。
    
    参数:
        AA_sequence (list or array): 氨基酸序列列表。
        model (keras.Model, optional): 已加载的模型对象。如果为 None，则从 model_file 加载。
        model_file (str, optional): 模型文件路径 (.h5)。
        
    返回:
        pd.DataFrame: 包含序列和预测活性值的 DataFrame。
    """
    
    if model == None:
        print(f"正在加载模型: {model_file}")
        model = tf.keras.models.load_model(model_file)
        
    # 将序列转换为 One-hot 编码 (注意: seq_to_onehot 返回 (onehot, integer)，取 [0] 是 onehot)
    X = seq_to_onehot(AA_sequence)[0]
    
    fitness_predictions = pd.DataFrame(AA_sequence, columns=['AA_sequence'])
    fitness_predictions['predicted_fitness'] = model.predict(X)
    
    return fitness_predictions
