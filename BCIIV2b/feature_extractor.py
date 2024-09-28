# 特征提取器
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import scipy.io as sio
import numpy as np

class EEGFeatureExtractor(nn.Module):
    def __init__(self, num_electrodes):
        super(EEGFeatureExtractor, self).__init__()
        
        # 时间卷积：沿时间轴的一维卷积
        self.temporal_conv = nn.Conv1d(in_channels=num_electrodes, 
                                       out_channels=60, 
                                       kernel_size=25, 
                                       stride=1)
        
        # 空间卷积：沿电极轴的一维卷积
        self.spatial_conv = nn.Conv1d(in_channels=60, 
                                      out_channels=60, 
                                      kernel_size=num_electrodes, 
                                      stride=1)
        
        # 批量标准化 Batch Normalization
        self.bn = nn.BatchNorm1d(60)
        
        # 平均池化：随着时间的推移池化 Average Pooling
        self.avg_pool = nn.AvgPool1d(kernel_size=75, stride=15)
        
        # Dropout
        self.dropout = nn.Dropout(p=0.5)


    def forward(self, x):
        # 时间卷积
        x = self.temporal_conv(x)
        
        # 空间卷积
        x = self.spatial_conv(x)
        
        # 批量标准化
        x = self.bn(x)
        
        # 平方激活
        x = x ** 2
        
        # 平均池化
        x = self.avg_pool(x)
        
        # 对数激活
        x = torch.log(x + 1e-6)  # 避免 x = 0 时取 log(0) 
         
        # Dropout
        x = self.dropout(x)
        # print(x.size())
        
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
        
        # 提取数据和标签，数据字段名为 'data' 和 'label'
        data = mat_data['data']  # shape: (320, 3, 1000)
        labels = mat_data['label'].flatten()  # 展平标签

        # 对每个样本进行特征提取
        for sample_idx in range(data.shape[0]):  # 遍历每个样本
            input_data = torch.tensor(data[sample_idx], dtype=torch.float32)  # shape: (3, 1000)
            input_data = input_data.unsqueeze(0)  # 增加批次维度，形状变为 (1, 3, 1000)

            # 前向传播
            output = model(input_data)
            
            # 转换为numpy数组以便于保存
            output_np = output.detach().numpy()  # 将输出转换为 numpy 数组

            # 构建新的文件名
            output_file_name = f"{file.split('.')[0]}_sample_{sample_idx}.mat"

            # 保存输出和标签到新的.mat文件
            sio.savemat(os.path.join(filtered_data_folder, output_file_name), 
                        {'extracted_features': output_np, 'label': labels[sample_idx]})

# 保存模型
model_path = "BCIIV2b/feature_extractor.pth"
torch.save(model.state_dict(), model_path)

print("模型已保存！")
print("特征提取完成！")
# FeatureExtractor = EEGFeatureExtractor(3)
# print(FeatureExtractor)



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


# # 划分训练集与测试集
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import scipy.io as sio
# import os
# import numpy as np
# from sklearn.model_selection import train_test_split
# from classifier import EEGClassifier
# # 自定义数据集类，用于加载.mat文件中的特征数据
# class BCI_Dataset(Dataset):
#     def __init__(self, data_dir):
#         self.features, self.labels = self.load_data(data_dir)

#     def load_data(self, data_dir):
#         features = []
#         labels = []
        
#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.mat'):
#                 mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#                 features.append(mat_data['extracted_features'])
#                 labels.append(mat_data['label'].flatten())  # 假设标签是以列的形式出现

#         features = np.concatenate(features, axis=0)  # 将特征合并为一个大的数组
#         labels = np.concatenate(labels, axis=0)  # 合并标签
#         return features, labels

#     def __len__(self):
#         return len(self.features)

#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # 特征提取器
# class EEGFeatureExtractor(nn.Module):
#     def __init__(self, num_electrodes):
#         super(EEGFeatureExtractor, self).__init__()
#         self.temporal_conv = nn.Conv1d(in_channels=num_electrodes, out_channels=60, kernel_size=25, stride=1)
#         self.spatial_conv = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=num_electrodes, stride=1)
#         self.bn = nn.BatchNorm1d(60)
#         self.avg_pool = nn.AvgPool1d(kernel_size=75, stride=15)
#         self.dropout = nn.Dropout(p=0.5)

#     def forward(self, x):
#         x = self.temporal_conv(x)
#         x = self.spatial_conv(x)
#         x = self.bn(x)
#         x = x ** 2
#         x = self.avg_pool(x)
#         x = torch.log(x + 1e-6)  # 添加epsilon以避免log(0)
#         x = self.dropout(x)
#         return x

# # 数据文件夹路径
# data_folder = 'data/feature_extractor_BCIIV2b_mat'

# # 创建数据集
# dataset = BCI_Dataset(data_folder)

# # 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(dataset.features, dataset.labels, test_size=0.2, random_state=42)

# # 创建 DataLoader
# train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
# test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# # 初始化特征提取器和分类器
# num_electrodes = 3  # 输入的电极数量
# feature_extractor = EEGFeatureExtractor(num_electrodes=num_electrodes)
# classifier = EEGClassifier(input_size=60, num_classes=2)

# # 优化器和损失函数
# optimizer = optim.Adam(classifier.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss()

# # 训练模型
# best_val_loss = float('inf')  # 初始化最好的验证损失
# for epoch in range(25):  # 假设训练25个epochs
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     # 训练过程
#     feature_extractor.train()
#     for inputs, labels in train_loader:
#         inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, 通道, 时间点)
        
#         features = feature_extractor(inputs)  # 特征提取
#         outputs = classifier(features)  # 分类输出

#         loss = criterion(outputs, labels)  # 计算损失
        
#         optimizer.zero_grad()  # 梯度清零
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
        
#         running_loss += loss.item() * inputs.size(0)
#         _, predicted = torch.max(outputs, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     epoch_loss = running_loss / total  # 当前 epoch 的平均损失
#     accuracy = correct / total * 100  # 计算模型的准确率
    
#     print(f'Epoch {epoch + 1}/{25}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

#     # 验证过程
#     feature_extractor.eval()  # 切换到评估模式
#     val_loss = 0.0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.permute(0, 2, 1)
#             features = feature_extractor(inputs)
#             outputs = classifier(features)
#             val_loss += criterion(outputs, labels).item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     val_loss /= total  # 当前平均验证损失
#     val_accuracy = correct / total * 100  # 当前平均验证准确率

#     print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

#     # 保存最好的模型
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(classifier.state_dict(), 'best_classifier.pth')  # 保存最好的模型

# # 测试模型
# def test_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
    
#     with torch.no_grad():
#         for inputs, labels in test_loader:
#             inputs = inputs.permute(0, 2, 1)
#             features = feature_extractor(inputs)  # 提取特征
#             outputs = classifier(features)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = correct / total * 100
#     print(f'Test Accuracy: {accuracy:.2f}%')

# # 调用测试模型
# test_model(classifier, test_loader)