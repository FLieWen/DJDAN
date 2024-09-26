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
        
#         # 遍历目录中的所有.mat文件
#         for file_name in os.listdir(data_dir):
#             if file_name.endswith('.mat'):
#                 mat_data = sio.loadmat(os.path.join(data_dir, file_name))
#                 # .mat文件中有'extracted_features'和'labels'键
#                 features.append(mat_data['extracted_features'])
#                 labels.append(mat_data['label'])
        
#         # 将列表拼接成numpy数组
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
#         # 定义一个1D卷积层，C-Conv，输出的类别数量是num_classes
#         self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=60)
#         # 使用Softmax激活函数将输出转为概率
#         self.softmax = nn.Softmax(dim=1)
    
#     # 前向传播函数
#     def forward(self, x):
#         # print("aaa",x.size())
#         x = self.conv1(x)
#         x = self.softmax(x)
#         # print(x.size())
#         return x

# # 定义训练函数
# def train_model(model, dataloader, criterion, optimizer, num_epochs=25):
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         for inputs, labels in dataloader:
#             inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, 通道, 时间点)
#             optimizer.zero_grad()  # 梯度清零
            
#             outputs = model(inputs)  # 前向传播
#             loss = criterion(outputs, labels)  # 计算损失
            
#             loss.backward()  # 反向传播
#             optimizer.step()  # 优化器更新参数
            
#             running_loss += loss.item() * inputs.size(0)  #  累加损失
#             _, predicted = torch.max(outputs, 1)  #  计算每个样本的预测类别
#             total += labels.size(0)  #  累加样本总数
#             correct += (predicted == labels).sum().item()  #  计算当前批次中正确预测的样本数量
        
#         epoch_loss = running_loss / total  #  计算当前 epoch 的平均损失
#         accuracy = correct / total * 100  #  计算模型的准确率
        
#         print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# # 加载数据集
# data_dir = 'data/feature_extractor_BCIIV2b_mat'  # 你的特征文件路径
# dataset = BCI_Dataset(data_dir)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# # 初始化模型、损失函数和优化器
# input_size = 60  # 输入的特征维度
# num_classes = 2  # 二分类问题
# model = EEGClassifier(input_size, num_classes)

# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# # 训练模型
# train_model(model, dataloader, criterion, optimizer, num_epochs=25)

# # 保存模型
# torch.save(model.state_dict(), 'classifier.pth')
# # classifier = EEGClassifier(61,2)
# # print(classifier)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import os
import numpy as np
from sklearn.model_selection import train_test_split

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
                # .mat文件中有'extracted_features'和'labels'键
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
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=60)
        # 使用Softmax激活函数将输出转为概率
        self.softmax = nn.Softmax(dim=1)
    
    # 前向传播函数
    def forward(self, x):
        # print("aaa",x.size())
        x = self.conv1(x)
        x = self.softmax(x)
        # print(x.size())
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
            
            running_loss += loss.item() * inputs.size(0)  #  累加损失
            _, predicted = torch.max(outputs, 1)  #  计算每个样本的预测类别
            total += labels.size(0)  #  累加样本总数
            correct += (predicted == labels).sum().item()  #  计算当前批次中正确预测的样本数量
        
        epoch_loss = running_loss / total  #  计算当前 epoch 的平均损失
        accuracy = correct / total * 100  #  计算模型的准确率
        
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')

# 加载数据集
data_dir = 'data/feature_extractor_BCIIV2b_mat'  # 你的特征文件路径
dataset = BCI_Dataset(data_dir)

# 使用 train_test_split 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(dataset.features, dataset.labels, test_size=0.2, random_state=42)

# 创建 DataLoader
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 初始化模型、损失函数和优化器
input_size = 60  # 输入的特征维度
num_classes = 2  # 二分类问题
model = EEGClassifier(input_size, num_classes)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练模型
train_model(model, train_loader, criterion, optimizer, num_epochs=25)

# 测试模型
def test_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.permute(0, 2, 1)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total * 100
    print(f'Test Accuracy: {accuracy:.2f}%')

# 测试模型
test_model(model, test_loader)

# 保存模型
torch.save(model.state_dict(), 'classifier.pth')
# classifier = EEGClassifier(61,2)
# print(classifier)
