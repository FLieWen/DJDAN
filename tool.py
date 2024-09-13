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
# file_path = 'data/feature_extractor_BCIIV2b_mat/B01E_features.mat'  # 替换为您的文件路径
# mat_data = sio.loadmat(file_path)

# # 查看.mat文件中所有的键
# print("Keys in the .mat file:", mat_data.keys())

# # 检查data和label的维度
# data = mat_data['extracted_features']  # 替换为实际的字段名
# label = mat_data['label']  # 替换为实际的字段名

# # 打印维度
# print("Shape of data:", data.shape)
# print("Shape of label:", label.shape)