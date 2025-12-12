"""
通用操作工具模块
================
此模块包含一些通用的数据操作函数，例如数据归一化。
"""

import numpy as np
import pandas as pd


def normalize(data,
              new_min=0,
              new_max=1
             ):
    """
    将数据归一化到指定范围 [new_min, new_max]。
    公式: new_val = (val - min) / (max - min) * (new_max - new_min) + new_min
    
    参数:
        data (pd.Series or np.array): 输入数据。
        new_min (float): 目标范围最小值。
        new_max (float): 目标范围最大值。
        
    返回:
        归一化后的数据。
    """
    
    old_range = data.max() - data.min() 
    new_range = new_max - new_min  
    new_data = (((data - data.min()) * new_range) / old_range) + new_min
    return new_data
    
#     a = (vmax - vmin) / (data.max()-data.min())
#     b = data.max() - a * data.max()
    
#     return a * data + b