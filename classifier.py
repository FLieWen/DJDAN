import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import os
import numpy as np

# 定义一个自定义的数据集类，用于加载.mat文件中的特征数据
class BCI_Dataset(Dataset):
    def __init__(self, data_dir):
        self.features, self.labels = self.load_data(data_dir)
    
    # 从.mat文件中加载数据
    def load_data(self, data_dir):
        features = []
        labels = []
        
        # 遍历data_dir中的所有.mat文件
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.mat'):
                mat_data = sio.loadmat(os.path.join(data_dir, file_name))
                # 假设.mat文件中有'features'和'labels'键
                features.append(mat_data['extracted_features'])
                labels.append(mat_data['label'])
        
        # 将数据列表拼接成numpy数组
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        return features, labels
    
    # 获取数据集的长度
    def __len__(self):
        return len(self.features)
    
    # 根据索引获取单个数据点
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# 定义分类器模型
class EEGClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGClassifier, self).__init__()
        # 定义一个1D卷积层，类似于论文中描述的分类器结构
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=61)
        self.softmax = nn.Softmax(dim=1)  # 使用softmax激活输出类别概率
    
    # 前向传播
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
            inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, channels, time)
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
data_dir = 'data/E_BCIIV2b_mat'  # 请根据你的实际路径调整
dataset = BCI_Dataset(data_dir)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
# input_size = dataset.features.shape[1]  # 输入的特征维度
input_size = 40  # 输入的特征维度
num_classes = 2  # 假设二分类问题
model = EEGClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器

# 训练模型
train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# 保存模型
torch.save(model.state_dict(), 'eeg_classifier.pth')


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