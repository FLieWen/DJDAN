# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import scipy.io as sio
# import os
# import numpy as np

# # 定义一个自定义的数据集类，用于加载.mat文件中的特征数据
# class BCI_Dataset(Dataset):
#     def __init__(self, data_dir):
#         self.features, self.labels = self.load_data(data_dir)
    
#     # 从.mat文件中加载数据
#     def load_data(self, data_dir):
#         features = []
#         labels = []
        
#         # 遍历data_dir中的所有.mat文件
#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.mat'):
#                 mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#                 # 假设.mat文件中有'features'和'labels'键
#                 features.append(mat_data['extracted_features'])
#                 labels.append(mat_data['label'])
        
#         # 将数据列表拼接成numpy数组
#         features = np.concatenate(features, axis=0)
#         labels = np.concatenate(labels, axis=0)
        
#         return features, labels
    
#     # 获取数据集的长度
#     def __len__(self):
#         return len(self.features)
    
#     # 根据索引获取单个数据点
#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # 定义分类器模型
# class EEGClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(EEGClassifier, self).__init__()
#         # 定义一个1D卷积层，类似于论文中描述的分类器结构
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=61)
#         self.softmax = nn.Softmax(dim=1)  # 使用softmax激活输出类别概率
    
#     # 前向传播
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.softmax(x)
#         return x

# # 定义训练函数
# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in dataloader:
#             inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, channels, time)
#             optimizer.zero_grad()  # 梯度清零
            
#             outputs = model(inputs)  # 前向传播
#             loss = criterion(outputs, labels)  # 计算损失
            
#             loss.backward()  # 反向传播
#             optimizer.step()  # 优化器更新参数
            
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / total
#         accuracy = correct / total * 100
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# # 加载数据集
# data_dir = 'data/E_BCIIV2b_mat'  # 请根据你的实际路径调整
# dataset = BCI_Dataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # 初始化模型、损失函数和优化器
# # input_size = dataset.features.shape[1]  # 输入的特征维度
# input_size = 40  # 输入的特征维度
# num_classes = 2  # 假设二分类问题
# model = EEGClassifier(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# # 训练模型
# train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# # 保存模型
# torch.save(model.state_dict(), 'eeg_classifier.pth')


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import scipy.io as sio
# import os
# import numpy as np

# # 定义一个自定义的数据集类，用于加载.mat文件中的特征数据
# class BCI_Dataset(Dataset):
#     def __init__(self, data_dir):
#         self.features, self.labels = self.load_data(data_dir)
    
#     # 从.mat文件中加载数据
#     def load_data(self, data_dir):
#         features = []
#         labels = []
        
#         # 遍历data_dir中的所有.mat文件
#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.mat'):
#                 mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#                 # 假设.mat文件中有'features'和'labels'键
#                 features.append(mat_data['extracted_features'])
#                 labels.append(mat_data['label'])
        
#         # 将数据列表拼接成numpy数组
#         features = np.concatenate(features, axis=0)
#         labels = np.concatenate(labels, axis=0)
        
#         return features, labels
    
#     # 获取数据集的长度
#     def __len__(self):
#         return len(self.features)
    
#     # 根据索引获取单个数据点
#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # 定义分类器模型
# class EEGClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(EEGClassifier, self).__init__()
#         # 定义一个1D卷积层，类似于论文中描述的分类器结构
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=61)
#         self.softmax = nn.Softmax(dim=1)  # 使用softmax激活输出类别概率
    
#     # 前向传播
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.softmax(x)
#         return x

# # 定义训练函数
# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in dataloader:
#             inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, channels, time)
#             optimizer.zero_grad()  # 梯度清零
            
#             outputs = model(inputs)  # 前向传播
#             loss = criterion(outputs, labels)  # 计算损失
            
#             loss.backward()  # 反向传播
#             optimizer.step()  # 优化器更新参数
            
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / total
#         accuracy = correct / total * 100
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# # 加载数据集
# data_dir = 'data/feature_extractor_BCIIV2b_mat'  # 请根据你的实际路径调整
# dataset = BCI_Dataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # 初始化模型、损失函数和优化器
# input_size = dataset.features.shape[1]  # 输入的特征维度
# num_classes = 2  # 假设二分类问题
# model = EEGClassifier(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# # 训练模型
# train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# # 保存模型
# torch.save(model.state_dict(), 'eeg_classifier.pth')




# 0915
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import os
import numpy as np

# 自定义数据集类，用于加载.mat文件中的特征数据
class BCI_Dataset(Dataset):
    def __init__(self, data_dir):
        self.features, self.labels = self.load_data(data_dir)
    
    # 从.mat文件中加载数据
    def load_data(self, data_dir):
        features = []
        labels = []
        
        # 遍历目录中的所有.mat文件
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.mat'):
                mat_data = sio.loadmat(os.path.join(data_dir, file_name))
                # 假设.mat文件中有'features'和'labels'键
                features.append(mat_data['extracted_features'])
                labels.append(mat_data['label'])
        
        # 将列表拼接成numpy数组
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    # 返回数据集的长度
    def __len__(self):
        return len(self.features)
    
    # 根据索引获取单个样本
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义分类器模型
class EEGClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGClassifier, self).__init__()
        # 定义一个1D卷积层，C-Conv，输出的类别数量是num_classes
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=40)
        # 使用Softmax激活函数将输出转为概率
        self.softmax = nn.Softmax(dim=1)
    
    # 前向传播函数
    def forward(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        return x

# 定义训练函数
def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, 通道, 时间点)
            optimizer.zero_grad()  # 梯度清零
            
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            
            loss.backward()  # 反向传播
            optimizer.step()  # 优化器更新参数
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / total
        accuracy = correct / total * 100
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 加载数据集
data_dir = 'data/T_feature_extractor_BCIIV2b_mat'  # 你的特征文件路径
dataset = BCI_Dataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
input_size = 60  # 输入的特征维度
num_classes = 2  # 二分类问题
model = EEGClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # 使用Adam优化器

# 训练模型
train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'T_classifier.pth')

# 使用训练后的模型进行预测
model.eval()  # 切换到评估模式

# 定义输出文件夹路径
output_dir = 'data/T_predictions'

# 检查并创建文件夹
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# all_predictions = []

# for inputs, labels in dataloader:
#     inputs = inputs.permute(0, 2, 1)
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs, 1)
#     all_predictions.extend(predicted.numpy())  # 收集预测结果

# # 保存预测结果到 .mat 文件
# output_filename = 'predictions.mat'
# sio.savemat(output_filename, {'predictions': np.array(all_predictions)})
# 对于 dataset 中的每个样本单独进行预测并保存
for idx in range(len(dataset)):
    inputs, labels = dataset[idx]
    inputs = inputs.unsqueeze(0).permute(0, 2, 1)  # 变形为 (1, 输入通道数, 时间点数)
    
    with torch.no_grad():  # 不计算梯度，进行评估
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
    # 保存每个样本的预测到新文件夹
    output_filename = os.path.join(output_dir, f'prediction_sample_{idx}.mat')
    sio.savemat(output_filename, {'prediction': predicted.numpy(), 'label': labels.numpy()})

print("模型训练和预测完成！")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import scipy.io as sio
# import os
# import numpy as np

# # 自定义数据集类，用于加载.mat文件中的特征数据
# class BCI_Dataset(Dataset):
#     def __init__(self, data_dir):
#         self.features, self.labels = self.load_data(data_dir)
    
#     # 从.mat文件中加载数据
#     def load_data(self, data_dir):
#         features = []
#         labels = []
        
#         max_label_length = 400  # 设定一个统一的标签长度

#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.mat'):
#                 mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#                 feature = mat_data['extracted_features']
#                 label = mat_data['label']
            
#                 # 统一标签大小（补零或者截断）
#                 if label.shape[0] < max_label_length:
#                     label = np.pad(label, (0, max_label_length - label.shape[0]), 'constant')
#                 elif label.shape[0] > max_label_length:
#                     label = label[:max_label_length]
            
#                 features.append(feature)
#                 labels.append(label)

#         # # 遍历目录中的所有.mat文件
#         # for file_name in os.listdir(data_dir):
#         #     if file_name.endswith('.mat'):
#         #         mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#         #         features.append(mat_data['extracted_features'])  # 提取特征
#         #         labels.append(mat_data['label'])  # 提取标签
        
#         features = np.concatenate(features, axis=0)
#         labels = np.concatenate(labels, axis=0)

#         return features, labels
    
#     # 返回数据集的长度
#     def __len__(self):
#         return len(self.features)
    
#     # 根据索引获取单个样本
#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# # 定义分类器模型
# class EEGClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(EEGClassifier, self).__init__()
#         # 定义一个1D卷积层，输入维度为40，卷积核大小为 (1, 61)
#         self.conv1 = nn.Conv1d(in_channels=40, out_channels=num_classes, kernel_size=61)
#         # 使用Softmax激活函数将输出转为概率
#         self.softmax = nn.Softmax(dim=1)
    
#     def forward(self, x):
#         x = self.conv1(x)  # 卷积层
#         x = x.squeeze(2)   # 去掉维度为1的通道
#         x = self.softmax(x)  # Softmax激活
#         return x

# # 加载数据集
# data_dir = 'data/E_feature_extractor_BCIIV2b_mat'
# dataset = BCI_Dataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # 初始化模型、损失函数和优化器
# input_size = 40  # 特征维度
# num_classes = 2  # 类别数量
# model = EEGClassifier(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()  # 交叉熵损失
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# # 训练模型
# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in dataloader:
#             inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, 通道, 时间点)
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / total
#         accuracy = correct / total * 100
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# train_model(model, dataloader, criterion, optimizer)

# # 保存模型
# torch.save(model.state_dict(), 'E_classifier.pth')
# print("模型训练完成，已保存到'E_classifier.pth'")



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# import scipy.io as sio
# import os
# import numpy as np

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
#                 labels.append(mat_data['label'].flatten())  # 确保标签是平坦的
        
#         features = np.concatenate(features, axis=0)
#         labels = np.concatenate(labels, axis=0)
        
#         return features, labels
    
#     def __len__(self):
#         return len(self.features)
    
#     def __getitem__(self, idx):
#         feature = self.features[idx]
#         label = self.labels[idx]
#         return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# class EEGClassifier(nn.Module):
#     def __init__(self, input_size, num_classes):
#         super(EEGClassifier, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=40)

#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in dataloader:
#             inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, num_channels, num_timepoints)
#             optimizer.zero_grad()
            
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
            
#             loss.backward()
#             optimizer.step()
            
#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
        
#         epoch_loss = running_loss / total
#         accuracy = correct / total * 100
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# # 加载数据集
# data_dir = 'data/feature_extractor_BCIIV2b_mat'
# dataset = BCI_Dataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # 初始化模型、损失函数和优化器
# input_size = 60  # assuming features are in (num_samples, num_features)
# num_classes = 2  # dynamically set the number of classes
# model = EEGClassifier(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)

# # 训练模型
# train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# # 保存模型
# torch.save(model.state_dict(), 'eeg_classifier.pth')

# # 使用训练后的模型进行预测
# model.eval()  # 切换到评估模式
# all_predictions = []

# for inputs, labels in dataloader:
#     inputs = inputs.permute(0, 2, 1)
#     outputs = model(inputs)
#     _, predicted = torch.max(outputs, 1)
#     all_predictions.extend(predicted.numpy())  # 收集预测结果

# # 保存预测结果到 .mat 文件
# output_filename = 'predictions.mat'
# sio.savemat(output_filename, {'predictions': np.array(all_predictions)})

# print("模型训练和预测完成！")