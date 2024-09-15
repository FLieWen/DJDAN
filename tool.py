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


import scipy.io as sio

# 加载.mat文件
file_path = 'data/T_predictions/prediction_sample_542.mat'  # 替换为您的文件路径
mat_data = sio.loadmat(file_path)

# 查看.mat文件中所有的键
print("Keys in the .mat file:", mat_data.keys())

# 检查data和label的维度
data = mat_data['prediction']  # 替换为实际的字段名
label = mat_data['label']  # 替换为实际的字段名

# 打印维度
print("Shape of data:", data.shape)
print("Shape of label:", label.shape)


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
