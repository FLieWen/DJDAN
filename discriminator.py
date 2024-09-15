import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import scipy.io as sio
import os
import numpy as np

# 加载预训练的特征提取器
class EEGFeatureExtractor(nn.Module):
    def __init__(self, num_electrodes):
        super(EEGFeatureExtractor, self).__init__()
        
        # Temporal Convolution: 1D convolution along the time axis
        self.temporal_conv = nn.Conv1d(in_channels=num_electrodes, 
                                       out_channels=40, 
                                       kernel_size=25, 
                                       stride=1)
        
        # Spatial Convolution: 1D convolution along the electrode axis
        self.spatial_conv = nn.Conv1d(in_channels=40, 
                                      out_channels=40, 
                                      kernel_size=num_electrodes, 
                                      stride=1)
        
        # Batch Normalization
        self.bn = nn.BatchNorm1d(40)
        
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
        
        return x

feature_extractor = EEGFeatureExtractor(num_electrodes=3)
feature_extractor.load_state_dict(torch.load('feature_extractor.pth'))
feature_extractor.eval()  # 切换到评估模式

# 加载预训练的分类器
class EEGClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGClassifier, self).__init__()
        # 定义一个1D卷积层，C-Conv，输出的类别数量是num_classes
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=61)
        # 使用Softmax激活函数将输出转为概率
        self.softmax = nn.Softmax(dim=1)
    
    # 前向传播函数
    def forward(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        return x

classifier = EEGClassifier(input_size=60, num_classes=2)
classifier.load_state_dict(torch.load('classifier.pth'))
classifier.eval()  # 切换到评估模式

# 数据加载器的定义
class BCI_Dataset(Dataset):
    def __init__(self, data_dir):
        self.features, self.labels = self.load_data(data_dir)
    
    def load_data(self, data_dir):
        features = []
        labels = []
        for file_name in os.listdir(data_dir):
            if file_name.endswith('.mat'):
                mat_data = sio.loadmat(os.path.join(data_dir, file_name))
                features.append(mat_data['data'])  # 假设数据在 'data' 键中
                labels.append(mat_data['label'].flatten())  # 标签展平为一维数组
        features = np.concatenate(features, axis=0)
        labels = np.concatenate(labels, axis=0)
        return features, labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return torch.tensor(feature, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

source_data_dir = 'data/BCIIV2b_mat/Processed_T_BCIIV2b_mat'
target_data_dir = 'data/BCIIV2b_mat/Processed_E_BCIIV2b_mat'

source_dataset = BCI_Dataset(source_data_dir)
target_dataset = BCI_Dataset(target_data_dir)

source_loader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=32, shuffle=True)

# 全局鉴别器的定义
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.conv = nn.Conv1d(in_channels=40, out_channels=2, kernel_size=61)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x

global_discriminator = GlobalDiscriminator()

# 局部鉴别器的定义
class LocalDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(LocalDiscriminator, self).__init__()
        self.conv = nn.Conv1d(in_channels=40, out_channels=num_classes, kernel_size=61)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.softmax(x)
        return x

local_discriminator = LocalDiscriminator(num_classes=2)

# 损失函数和训练过程
criterion = nn.CrossEntropyLoss()
optimizer_global = optim.Adam(global_discriminator.parameters(), lr=0.001)
optimizer_local = optim.Adam(local_discriminator.parameters(), lr=0.001)

# GRL函数 (用于梯度反转层)
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

# 训练全局鉴别器和局部鉴别器
for epoch in range(10):  # 假设训练10个epoch
    for (source_data, _), (target_data, _) in zip(source_loader, target_loader):
        # 提取源域和目标域特征
        source_features = feature_extractor(source_data)
        target_features = feature_extractor(target_data)

        # 全局鉴别器
        domain_labels_source = torch.zeros(source_features.size(0)).long()  # 源域标签为0
        domain_labels_target = torch.ones(target_features.size(0)).long()  # 目标域标签为1

        # 反转梯度
        source_features_reversed = GradientReversalLayer.apply(source_features)
        target_features_reversed = GradientReversalLayer.apply(target_features)

        global_output_source = global_discriminator(source_features_reversed)
        global_output_target = global_discriminator(target_features_reversed)

        domain_labels_source = domain_labels_source.unsqueeze(1)
        domain_labels_target = domain_labels_target.unsqueeze(1)

        global_loss_source = criterion(global_output_source, domain_labels_source)
        global_loss_target = criterion(global_output_target, domain_labels_target)

        global_loss = global_loss_source + global_loss_target
        optimizer_global.zero_grad()
        global_loss.backward()
        optimizer_global.step()

        # 局部鉴别器
        source_preds = classifier(source_features)
        target_preds = classifier(target_features)

        combined_source = torch.cat((source_features, source_preds), dim=1)
        combined_target = torch.cat((target_features, target_preds), dim=1)

        local_output_source = local_discriminator(combined_source)
        local_output_target = local_discriminator(combined_target)

        local_loss_source = criterion(local_output_source, domain_labels_source)
        local_loss_target = criterion(local_output_target, domain_labels_target)

        local_loss = local_loss_source + local_loss_target
        optimizer_local.zero_grad()
        local_loss.backward()
        optimizer_local.step()

    print(f'Epoch {epoch+1}, Global Loss: {global_loss.item()}, Local Loss: {local_loss.item()}')