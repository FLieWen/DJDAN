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