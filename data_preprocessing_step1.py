# import numpy as np
# from scipy.signal import butter, filtfilt


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
#     fa = 0.5 * fs
#     low = lowcut / fa
#     high = highcut / fa
#     b, a = butter(order, [low, high], btype='band')
#     y = filtfilt(b, a, data)
#     return y


# # 生成噪声信号
# fs = 1000  # 采样率
# t = np.arange(0, 1, 1/fs)  # 时间序列

# x = 5*np.sin(2*np.pi*50*t) + 2*np.sin(2*np.pi*120*t) + 0.5*np.sin(2*np.pi*200*t)
# noise = 0.5*np.random.randn(len(t))
# y = x + noise

# # 设计巴特沃斯滤波器，滤除50 Hz以下和200 Hz以上的信号
# filtered_signal = butter_bandpass_filter(x, 4, 38, fs, order=3)

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
# fs = 1000  # 采样率
# lowcut = 4  # 低频截止
# highcut = 38  # 高频截止
# order = 3  # 滤波器的阶数

# # 定义文件夹路径
# data_folder = 'data/BCIIV2b_mat'
# filtered_data_folder = 'data/Filtered_BCIIV2b_mat'

# # 创建输出文件夹
# if not os.path.exists(filtered_data_folder):
#     os.makedirs(filtered_data_folder)

# # 处理每个mat文件
# for file in os.listdir(data_folder):
#     if file.endswith('.mat'):
#         # 加载.mat文件
#         mat_data = sio.loadmat(os.path.join(data_folder, file))
        
#         # 假设数据存储在名为 'data' 的字段中
#         data = mat_data['data']  # 根据您mat文件中的实际字段名进行调整
        
#         # 处理数据的每一列（假设数据是二维的）
#         filtered_data = []
#         for i in range(data.shape[1]):
#             filtered_signal = butter_bandpass_filter(data[:, i], lowcut, highcut, fs, order)
#             filtered_data.append(filtered_signal)
        
#         # 将过滤后的数据转换为数组
#         filtered_data = np.array(filtered_data).T  # 转置回原来的形状
        
#         # 保存过滤后的数据到新文件
#         sio.savemat(os.path.join(filtered_data_folder, file), {'filtered_data': filtered_data})

# print("数据处理完成！")

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
data_folder = 'data/BCIIV2b_mat'
filtered_data_folder = 'data/Filtered_BCIIV2b_mat'

# 创建输出文件夹
if not os.path.exists(filtered_data_folder):
    os.makedirs(filtered_data_folder)

# 处理每个mat文件
for file in os.listdir(data_folder):
    if file.endswith('.mat'):
        # 加载.mat文件
        mat_data = sio.loadmat(os.path.join(data_folder, file))
        
        # 假设数据存储在 'data' 和 'label' 的字段中
        data = mat_data['data'].flatten()  # 将数据展平为一维数组
        label = mat_data['label'].flatten()  # 将标签展平为一维数组，确保同样是一维
        
        # 应用巴特沃斯带通滤波器
        filtered_signal = butter_bandpass_filter(data, lowcut, highcut, fs, order)
        
        # 保存过滤后的数据和标签到新文件
        sio.savemat(os.path.join(filtered_data_folder, file), {'data': filtered_signal, 'label': label})

print("数据处理完成！")