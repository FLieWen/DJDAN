# 数据预处理：三阶巴特沃斯带通滤波器
import numpy as np
import os
import scipy.io as sio
from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    fa = 0.5 * fs
    low = lowcut / fa
    high = highcut / fa
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 设置参数
fs = 1000  # 采样率
lowcut = 4  # 低频截止
highcut = 38  # 高频截止
order = 3  # 滤波器的阶数

# 定义文件夹路径
data_folder = 'data/new_BCIIV2b_mat'
filtered_data_folder = 'data/Filtered_BCIIV2b_mat'

# 创建输出文件夹
if not os.path.exists(filtered_data_folder):
    os.makedirs(filtered_data_folder)

# 处理每个mat文件
for file in os.listdir(data_folder):
    if file.endswith('.mat'):
        # 加载.mat文件
        mat_data = sio.loadmat(os.path.join(data_folder, file))
        
        # 数据存储在 'data' 和 'label' 的字段中
        data = mat_data['data']  # shape: (320, 3, 1000)
        labels = mat_data['label'].flatten()  # 标签
        num_samples, num_electrodes, num_timepoints = data.shape
        
        # 创建一个用于存储过滤后数据的数组
        filtered_data = np.zeros((num_samples, num_electrodes, num_timepoints))
        
        # 对每个样本和每个电极进行滤波
        for sample_idx in range(num_samples):
            for electrode_idx in range(num_electrodes):
                # 选择单个电极的信号
                single_channel_data = data[sample_idx, electrode_idx, :]  
                # 应用巴特沃斯带通滤波器
                filtered_signal = butter_bandpass_filter(single_channel_data, lowcut, highcut, fs, order)
                # 存储过滤后的信号
                filtered_data[sample_idx, electrode_idx, :] = filtered_signal

        # 保存过滤后的数据和标签到新文件
        sio.savemat(os.path.join(filtered_data_folder, file), {'data': filtered_data, 'label': labels})

print("数据处理完成！")
