# # 数据预处理：三阶巴特沃斯带通滤波器
# import numpy as np
# import os
# import scipy.io as sio
# from scipy.signal import butter, filtfilt

# def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
#     fa = 0.5 * fs
#     low = lowcut / fa
#     high = highcut / fa
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data)
#     return y

# # 设置参数
# fs = 250  # 采样率
# lowcut = 4  # 低频截止
# highcut = 38  # 高频截止
# order = 3  # 滤波器的阶数

# # 定义文件夹路径
# data_folder = 'data/new_BCIIV2a_mat'
# filtered_data_folder = 'data/Filtered_BCIIV2a_mat'

# # 创建输出文件夹
# if not os.path.exists(filtered_data_folder):
#     os.makedirs(filtered_data_folder)

# # 处理每个mat文件
# for file in os.listdir(data_folder):
#     if file.endswith('.mat'):
#         # 加载.mat文件
#         mat_data = sio.loadmat(os.path.join(data_folder, file))
        
#         # 数据存储在 'data' 和 'label' 的字段中
#         data = mat_data['data']  # shape: (320, 3, 1000)
#         labels = mat_data['label'].flatten()  # 标签
#         num_samples, num_electrodes, num_timepoints = data.shape
        
#         # 创建一个用于存储过滤后数据的数组
#         filtered_data = np.zeros((num_samples, num_electrodes, num_timepoints))
        
#         # 对每个样本和每个电极进行滤波
#         for sample_idx in range(num_samples):
#             for electrode_idx in range(num_electrodes):
#                 # 选择单个电极的信号
#                 single_channel_data = data[sample_idx, electrode_idx, :]  
#                 # 应用巴特沃斯带通滤波器
#                 filtered_signal = butter_bandpass_filter(single_channel_data, lowcut, highcut, fs, order)
#                 # 存储过滤后的信号
#                 filtered_data[sample_idx, electrode_idx, :] = filtered_signal

#         # 保存过滤后的数据和标签到新文件
#         sio.savemat(os.path.join(filtered_data_folder, file), {'data': filtered_data, 'label': labels})

# print("数据处理完成！")


import numpy as np
import os
import mne
from scipy.signal import butter, filtfilt

# 巴特沃斯带通滤波器
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    fa = 0.5 * fs
    low = lowcut / fa
    high = highcut / fa
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

# 设置参数
fs = 250  # 采样率 (根据具体数据集进行调整)
lowcut = 4  # 低频截止
highcut = 38  # 高频截止
order = 3  # 滤波器阶数

# 定义文件夹路径
data_folder = 'data/new_BCIIV2a_gdf'  # 你可以将GDF文件放在此文件夹
filtered_data_folder = 'data/Filtered_BCIIV2a_gdf'

# 创建输出文件夹
if not os.path.exists(filtered_data_folder):
    os.makedirs(filtered_data_folder)

# 处理每个.gdf文件
for file in os.listdir(data_folder):
    if file.endswith('.gdf'):
        # 加载GDF文件
        raw = mne.io.read_raw_gdf(os.path.join(data_folder, file), preload=True)
        
        # 提取数据和标签
        data = raw.get_data()  # data.shape: (n_channels, n_times)
        events, event_ids = mne.events_from_annotations(raw)  # 提取事件（标签）
        
        num_electrodes, num_timepoints = data.shape
        num_samples = len(events)
        
        # 创建一个用于存储过滤后数据的数组
        filtered_data = np.zeros((num_samples, num_electrodes, num_timepoints))
        
        # 对每个样本和每个电极进行滤波
        for electrode_idx in range(num_electrodes):
            # 获取每个通道的信号
            single_channel_data = data[electrode_idx, :]
            # 应用巴特沃斯带通滤波器
            filtered_signal = butter_bandpass_filter(single_channel_data, lowcut, highcut, fs, order)
            # 存储过滤后的信号
            filtered_data[:, electrode_idx, :] = filtered_signal

        # 保存过滤后的数据和标签到新文件
        np.savez(os.path.join(filtered_data_folder, file.replace('.gdf', '.npz')), 
                 data=filtered_data, 
                 labels=events[:, 2])  # 使用事件作为标签

print("数据处理完成！")
