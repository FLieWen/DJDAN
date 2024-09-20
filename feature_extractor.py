import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.io as sio
import numpy as np

class EEGFeatureExtractor(nn.Module):
    def __init__(self, num_electrodes):
        super(EEGFeatureExtractor, self).__init__()
        
        # Temporal Convolution: 1D convolution along the time axis
        self.temporal_conv = nn.Conv1d(in_channels=num_electrodes, 
                                       out_channels=60, 
                                       kernel_size=25, 
                                       stride=1)
        
        # Spatial Convolution: 1D convolution along the electrode axis
        self.spatial_conv = nn.Conv1d(in_channels=60, 
                                      out_channels=60, 
                                      kernel_size=num_electrodes, 
                                      stride=1)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(60)
        
        # Average Pooling: Pooling over time
        self.avg_pool = nn.AvgPool1d(kernel_size=75, stride=15)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        # Temporal Convolution
        x = self.temporal_conv(x)
        
        # Spatial Convolution
        x = self.spatial_conv(x)
        
        # Batch Normalization
        x = self.bn(x)
        
        # Square Activation
        x = x ** 2
        
        # Average Pooling
        x = self.avg_pool(x)
        
        # Logarithm Activation
        x = torch.log(x + 1e-6)  # Adding epsilon to avoid log(0)
        
        # Dropout
        x = self.dropout(x)
        print(x.size())
        
        return x

# 定义数据文件夹路径
data_folder = 'data/BCIIV2b_mat/new_Processed_BCIIV2b_mat'
filtered_data_folder = 'data/feature_extractor_BCIIV2b_mat'

# 创建输出文件夹
if not os.path.exists(filtered_data_folder):
    os.makedirs(filtered_data_folder)

# 指定模型参数
num_electrodes = 3  # 使用的电极数量
model = EEGFeatureExtractor(num_electrodes=num_electrodes)

# 对文件进行处理
for file in os.listdir(data_folder):
    if file.endswith('.mat'):
        # 加载.mat文件
        mat_data = sio.loadmat(os.path.join(data_folder, file))
        
        # 提取数据和标签，假设数据字段名为 'data' 和 'label'
        data = mat_data['data']  # shape: (320, 3, 1000)
        labels = mat_data['label'].flatten()  # 展平标签

        # 对每个样本进行特征提取
        for sample_idx in range(data.shape[0]):  # 遍历每个样本
            input_data = torch.tensor(data[sample_idx], dtype=torch.float32)  # shape: (3, 1000)
            input_data = input_data.unsqueeze(0)  # 增加批次维度，形状变为 (1, 3, 1000)

            # Forward pass
            output = model(input_data)  # output shape will depend on the pooling and input dimensions
            
            # 转换为numpy数组以便于保存
            output_np = output.detach().numpy()  # 将输出转换为 numpy 数组

            # 构建新的文件名
            output_file_name = f"{file.split('.')[0]}_sample_{sample_idx}.mat"
            # 保存输出和标签到新的.mat文件
            sio.savemat(os.path.join(filtered_data_folder, output_file_name), 
                        {'extracted_features': output_np, 'label': labels[sample_idx]})

# 保存模型
model_path = "feature_extractor.pth"
torch.save(model.state_dict(), model_path)

print("模型已保存！")
print("特征提取完成！")
FeatureExtractor = EEGFeatureExtractor(3)
print(FeatureExtractor)




# # 0915
# import torch
# import torch.nn as nn
# import os
# import scipy.io as sio
# import numpy as np

# class EEGFeatureExtractor(nn.Module):
#     def __init__(self, num_electrodes):
#         super(EEGFeatureExtractor, self).__init__()

#         # 时间卷积：沿时间轴的一维卷积
#         self.temporal_conv = nn.Conv1d(in_channels=num_electrodes, 
#                                        out_channels=40, 
#                                        kernel_size=25, 
#                                        stride=1, 
#                                        padding=12)  # 添加 padding
        
#         # 空间卷积：沿电极轴的一维卷积
#         self.spatial_conv = nn.Conv1d(in_channels=40, 
#                                       out_channels=40, 
#                                       kernel_size=num_electrodes, 
#                                       stride=1)
        
#         # 批量标准化
#         self.bn = nn.BatchNorm1d(40)
        
#         # 平均池化
#         self.avg_pool = nn.AvgPool1d(kernel_size=75, stride=15)
        
#         # Dropout
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = self.temporal_conv(x)
#         x = self.spatial_conv(x)
#         x = self.bn(x)
#         x = x ** 2
#         x = self.avg_pool(x)
#         x = torch.log(x + 1e-6)  # Adding epsilon to avoid log(0)
#         x = self.dropout(x)
#         return x

# # 定义数据文件夹路径
# data_folder = 'data/T_Standardized_BCIIV2b_mat'
# filtered_data_folder = 'data/T_feature_extractor_BCIIV2b_mat'

# # 创建输出文件夹
# if not os.path.exists(filtered_data_folder):
#     os.makedirs(filtered_data_folder)

# # 指定模型参数
# num_electrodes = 3  # 使用的电极数量
# model = EEGFeatureExtractor(num_electrodes=num_electrodes)

# # 对文件进行处理
# for file in os.listdir(data_folder):
#     if file.endswith('.mat'):
#         # 加载.mat文件
#         mat_data = sio.loadmat(os.path.join(data_folder, file))
        
#         # 提取数据和标签
#         data = mat_data['data']  # shape: (320, 3, 1000)
#         labels = mat_data['label'].flatten()  # 展平标签

#         # 创建一个列表来存储所有样本的特征
#         all_features = []

#         # 对每个样本进行特征提取
#         for sample_idx in range(data.shape[0]):
#             input_data = torch.tensor(data[sample_idx], dtype=torch.float32)  # shape: (3, 1000)
#             input_data = input_data.unsqueeze(0)  # 增加批次维度，形状变为 (1, 3, 1000)

#             # Forward pass
#             output = model(input_data)  # output shape will depend on the pooling and input dimensions
            
#             # 转换为numpy数组以便于保存
#             output_np = output.detach().numpy()  # 将输出转换为 numpy 数组
            
#             # 将当前样本的特征添加到列表
#             all_features.append(output_np)

#         # 合并所有样本的特征
#         all_features_np = np.vstack(all_features)

#         # 保存合并后的特征和标签到新的.mat文件
#         output_file_name = f"{file.split('.')[0]}_features.mat"  # 生成输出文件名
#         sio.savemat(os.path.join(filtered_data_folder, output_file_name), 
#                     {'extracted_features': all_features_np, 'label': labels})

# # 保存训练好的特征提取器模型
# model_save_path = 'T_feature_extractor.pth'
# torch.save(model.state_dict(), model_save_path)

# print(f"特征提取完成，所有样本的特征已保存到各自的.mat文件中！")
# print(f"训练好的特征提取器模型已保存到 {model_save_path}")

