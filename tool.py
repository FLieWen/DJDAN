# import scipy.io as sio

# # 读取.mat文件
# mat_data = sio.loadmat('data\BCIIV2b_mat\B01E.mat')

# # 提取变量
# matrix1 = mat_data['data']
# # matrix2 = mat_data['matrix2']

# # 显示变量信息
# print("matrix1的形状:", matrix1.shape)
# # print("matrix2的数据类型:", type(matrix2))


# # 读取查看.mat文件
# from scipy.io import loadmat
# features_struct=loadmat( 'data\BCIIV2b_mat\B01E.mat' )
# print(features_struct)


# import scipy.io
# mat_data=scipy.io.loadmat('data\BCIIV2b_mat\B01E.mat')
# # 假设mat文件中有一个名为'my_matrix'的矩阵变量
# matrix_data = mat_data['data']
# #打印矩阵的维度
# print(matrix_data.shape)


# import scipy.io as sio

# # 加载.mat文件
# mat_data = sio.loadmat('data/Standardized_BCIIV2b_mat/B01E.mat')  # 替换为您的文件路径

# # 假设数据存储在 'data' 字段中
# data = mat_data['data']  # 读取数据

# # 打印数据的形状
# print(data.shape)  # (num_electrodes, time_points) 或 (time_points, num_electrodes)

# # 确定电极数
# if len(data.shape) == 2:
#     if data.shape[0] > data.shape[1]:
#         num_electrodes = data.shape[0]  # 第一维为电极数
#     else:
#         num_electrodes = data.shape[1]  # 第二维为电极数
# else:
#     raise ValueError("数据的形状不符合预期。")
    
# print(f"电极数量: {num_electrodes}")


# import scipy.io as sio

# # 加载.mat文件
# file_path = 'data/T_predictions/prediction_sample_542.mat'  # 替换为您的文件路径
# mat_data = sio.loadmat(file_path)

# # 查看.mat文件中所有的键
# print("Keys in the .mat file:", mat_data.keys())

# # 检查data和label的维度
# data = mat_data['prediction']  # 替换为实际的字段名
# label = mat_data['label']  # 替换为实际的字段名

# # 打印维度
# print("Shape of data:", data.shape)
# print("Shape of label:", label.shape)


# import scipy.io
# import numpy as np
# import os

# # 文件夹路径
# folder_path = 'data/E_feature_extractor_BCIIV2b_mat'

# # 获取所有 .mat 文件
# mat_files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

# # 目标形状
# target_shape = (320, 40, 62)

# # 函数用于调整数据形状，如果数据不匹配就进行填充或裁剪
# def pad_or_crop(data, target_shape):
#     current_shape = data.shape
#     padded_data = np.zeros(target_shape)
    
#     # 获取最小的shape
#     min_shape = tuple(min(cs, ts) for cs, ts in zip(current_shape, target_shape))
    
#     # 填充或裁剪数据
#     padded_data[:min_shape[0], :min_shape[1], :min_shape[2]] = data[:min_shape[0], :min_shape[1], :min_shape[2]]
    
#     return padded_data

# # 处理每个文件
# for mat_file in mat_files:
#     mat_data = scipy.io.loadmat(os.path.join(folder_path, mat_file))
    
#     # 这里假设 mat 文件中包含 'data' 键
#     if 'data' in mat_data:
#         data = mat_data['data']
#         print(f"Original shape of {mat_file}: {data.shape}")
        
#         # 调整形状
#         reshaped_data = pad_or_crop(data, target_shape)
#         print(f"Reshaped data of {mat_file}: {reshaped_data.shape}")
        
#         # 保存新的 .mat 文件
#         new_file_path = os.path.join(folder_path, 'reshaped_' + mat_file)
#         scipy.io.savemat(new_file_path, {'data': reshaped_data})
#         print(f"Saved reshaped data to {new_file_path}")
#     else:
#         print(f"No 'data' found in {mat_file}")


# import numpy as np
# import os
# import scipy.io as sio

# # 定义文件夹路径
# source_data_folder = 'data/BCIIV2b_mat/T_BCIIV2b_mat'  # 源域数据文件夹路径
# target_data_folder = 'data/BCIIV2b_mat/E_BCIIV2b_mat'  # 目标域数据文件夹路径
# output_source_folder = 'data/BCIIV2b_mat/Processed_T_BCIIV2b_mat'  # 处理后的源域数据文件夹
# output_target_folder = 'data/BCIIV2b_mat/Processed_E_BCIIV2b_mat'  # 处理后的目标域数据文件夹

# # 创建输出文件夹
# if not os.path.exists(output_source_folder):
#     os.makedirs(output_source_folder)

# if not os.path.exists(output_target_folder):
#     os.makedirs(output_target_folder)

# # 设置统一的实验数量
# source_experiment_size = 400  # 源域标准实验数
# target_experiment_size = 320  # 目标域标准实验数

# def process_data_and_labels(data, labels, required_size):
#     """处理数据和标签，将数据和标签扩充或截断到指定的大小"""
#     current_size = data.shape[0]  # 当前实验数量
    
#     if current_size < required_size:
#         # 实验数不足时，补全数据（填充零值）
#         padding_data = np.zeros((required_size - current_size,) + data.shape[1:])
#         processed_data = np.vstack((data, padding_data))
        
#         # 标签补全（选择填充最后一个标签）
#         padding_labels = np.full((required_size - current_size,), labels[-1])
#         processed_labels = np.concatenate((labels, padding_labels))
    
#     else:
#         # 实验数超出时，进行截断
#         processed_data = data[:required_size]
#         processed_labels = labels[:required_size]
    
#     return processed_data, processed_labels

# # 处理源域数据
# for file in os.listdir(source_data_folder):
#     if file.endswith('.mat'):
#         # 加载.mat文件
#         mat_data = sio.loadmat(os.path.join(source_data_folder, file))
#         data = mat_data['data']  # 假设数据存储在 'data' 键中
#         labels = mat_data['label'].flatten()  # 标签展平为一维数组
        
#         # 处理数据和标签
#         processed_data, processed_labels = process_data_and_labels(data, labels, source_experiment_size)
        
#         # 保存处理后的数据和标签到新文件
#         sio.savemat(os.path.join(output_source_folder, file), {'data': processed_data, 'label': processed_labels})

# # 处理目标域数据
# for file in os.listdir(target_data_folder):
#     if file.endswith('.mat'):
#         # 加载.mat文件
#         mat_data = sio.loadmat(os.path.join(target_data_folder, file))
#         data = mat_data['data']  # 假设数据存储在 'data' 键中
#         labels = mat_data['label'].flatten()  # 标签展平为一维数组
        
#         # 处理数据和标签
#         processed_data, processed_labels = process_data_and_labels(data, labels, target_experiment_size)
        
#         # 保存处理后的数据和标签到新文件
#         sio.savemat(os.path.join(output_target_folder, file), {'data': processed_data, 'label': processed_labels})

# print("源域和目标域数据处理完成！")

import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import os
import numpy as np
import torch.nn as nn

# 自定义 Dataset，用于加载 .mat 文件
class BCIIV2bDataset(Dataset):
    def __init__(self, mat_files_folder):
        self.mat_files_folder = mat_files_folder
        self.mat_files = [f for f in os.listdir(mat_files_folder) if f.endswith('.mat')]

    def __len__(self):
        return len(self.mat_files)

    def __getitem__(self, idx):
        # 加载 .mat 文件
        mat_file = self.mat_files[idx]
        mat_data = scipy.io.loadmat(os.path.join(self.mat_files_folder, mat_file))
        
        # 提取数据和标签 (根据你的 .mat 文件结构，假设 'data' 和 'labels' 是键名)
        data = mat_data['extracted_features']  # 信号数据
        label = mat_data['label']  # 标签
        
        # 转换数据类型为 torch.Tensor
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # 标签转换为 long 类型

        return data, label

# 计算类别数 C 的函数
def get_num_classes(data_loader):
    unique_labels = set()  # 用于存储唯一的标签
    for _, labels in data_loader:  # 遍历 DataLoader 获取标签
        unique_labels.update(labels.numpy().flatten())  # 提取标签并将其添加到集合中
    return len(unique_labels)

# 分类器定义
class Classifier(nn.Module):
    def __init__(self, C):
        super(Classifier, self).__init__()
        # C-Conv: 1 × 61 kernel, C output channels (number of classes)
        self.conv = nn.Conv1d(40, C, kernel_size=61)
        # Softmax
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x

# 主函数部分
if __name__ == "__main__":
    # 加载自定义数据集
    mat_folder_path = 'data/feature_extractor_BCIIV2b_mat'  # 设置 .mat 文件路径
    dataset = BCIIV2bDataset(mat_folder_path)
    
    # 使用 DataLoader 加载数据
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 计算类别数 C
    C = get_num_classes(data_loader)
    print(f"数据集中类别数 C: {C}")

    # 初始化分类器
    classifier = Classifier(C)
    print(classifier)