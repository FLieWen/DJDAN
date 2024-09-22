# 指数移动标准化
import numpy as np
import os
import scipy.io as sio

# 设置衰减因子
beta = 0.999

def exponential_moving_standardization(data, beta):
    """进行电极指数移动标准化"""
    # 初始化加权均值weighted_mean与加权方差weighted_var
    weighted_mean = 0
    weighted_var = 0
    
    # 获取数据形状
    num_samples, num_electrodes, num_timepoints = data.shape
    standardized_data = np.zeros_like(data)

    # 对每个样本和每个电极进行标准化
    for sample_idx in range(num_samples):
        for electrode_idx in range(num_electrodes):
            for i in range(num_timepoints):
                # 更新加权均值
                weighted_mean = beta * weighted_mean + (1 - beta) * data[sample_idx, electrode_idx, i]
                # 更新加权方差
                weighted_var = beta * weighted_var + (1 - beta) * (data[sample_idx, electrode_idx, i] - weighted_mean) ** 2
                
                # 计算加权标准差
                weighted_std = np.sqrt(weighted_var)

                # 标准化
                if weighted_std > 0:  # 为避免除以零
                    standardized_data[sample_idx, electrode_idx, i] = (data[sample_idx, electrode_idx, i] - weighted_mean) / weighted_std
                else:
                    standardized_data[sample_idx, electrode_idx, i] = 0  # 或者设置为某个固定值

    return standardized_data

# 定义文件夹路径
data_folder = 'data/Filtered_BCIIV2b_mat'
standardized_data_folder = 'data/Standardized_BCIIV2b_mat'

# 创建输出文件夹
if not os.path.exists(standardized_data_folder):
    os.makedirs(standardized_data_folder)

# 处理每个 .mat 文件
for file in os.listdir(data_folder):
    if file.endswith('.mat'):
        # 加载 .mat 文件
        mat_data = sio.loadmat(os.path.join(data_folder, file))

        # 数据存储在 'data' 和 'label' 字段中
        data = mat_data['data']  # shape: (320, 3, 1000)
        label = mat_data['label'].flatten()  # 将标签展平为一维数组
        
        # 进行电极指数移动标准化
        standardized_signal = exponential_moving_standardization(data, beta)
        
        # 保存标准化后的数据和标签到新文件
        sio.savemat(os.path.join(standardized_data_folder, file), {'data': standardized_signal, 'label': label})

print("数据处理完成！")
