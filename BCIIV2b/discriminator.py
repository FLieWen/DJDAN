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
        
        # 平均池化：随着时间的推移池化 Average Pooling: Pooling over time
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
        x = torch.log(x + 1e-6)  # Adding epsilon to avoid log(0)
        
        # Dropout
        x = self.dropout(x)
        # print(x.size())

        return x

feature_extractor = EEGFeatureExtractor(num_electrodes=3)
feature_extractor.load_state_dict(torch.load('BCIIV2b/feature_extractor.pth'))
feature_extractor.eval()  # 切换到评估模式

# 加载预训练的分类器
class EEGClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EEGClassifier, self).__init__()
        # 定义一个1D卷积层，C-Conv，输出的类别数量是num_classes
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=num_classes, kernel_size=60)
        # 使用Softmax激活函数将输出转为概率
        self.softmax = nn.Softmax(dim=1)
    
    # 前向传播函数
    def forward(self, x):
        # print("zzz",x.size())
        x = self.conv1(x)
        x = self.softmax(x)
        return x

classifier = EEGClassifier(input_size=60, num_classes=2)
classifier.load_state_dict(torch.load('BCIIV2b/best_classifier.pth'))
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
                features.append(mat_data['data'])  # 数据在 'data' 键中
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

source_loader = DataLoader(source_dataset, batch_size=64, shuffle=True)
target_loader = DataLoader(target_dataset, batch_size=64, shuffle=True)

# 全局鉴别器的定义
class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super(GlobalDiscriminator, self).__init__()
        self.conv = nn.Conv1d(in_channels=60, out_channels=1, kernel_size=60)
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 全局平均池化，输出1个值

    def forward(self, x):
        # print(x.size())
        x = self.conv(x)
        x = self.global_pool(x)  # 使用全局平均池化，将 (32, 1, 21) 转为 (32, 1, 1)
        x = x.view(x.size(0), -1)  # 将输出展平为 (32, 1)
        return x

global_discriminator = GlobalDiscriminator()

# 局部鉴别器的定义
class LocalDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(LocalDiscriminator, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=60, out_channels=1, kernel_size=60) for _ in range(num_classes)
        ])
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # 用于池化特征

    def forward(self, features, preds):
        outputs = []
        for i, conv in enumerate(self.convs):
            class_mask = preds[:, i].unsqueeze(2)  # 得到该类别的预测概率
            class_features = features * class_mask  # 特征与类别概率相乘
            output = conv(class_features)  # 通过该类别的卷积层
            output = self.global_pool(output).view(output.size(0), -1)  # 池化并展平输出
            outputs.append(output)
        return torch.cat(outputs, dim=1)  # 将每个类别的输出拼接起来
    # def __init__(self, num_classes):
    #     super(LocalDiscriminator, self).__init__()
    #     self.conv = nn.Conv1d(in_channels=40, out_channels=num_classes, kernel_size=40)
    #     self.softmax = nn.Softmax(dim=1)

    # def forward(self, x):
    #     x = self.conv(x)
    #     x = self.softmax(x)
    #     return x

local_discriminator = LocalDiscriminator(num_classes=1)

# 损失函数和训练过程
criterion = nn.BCEWithLogitsLoss()
optimizer_global = optim.Adam(global_discriminator.parameters(), lr=0.0005)
optimizer_local = optim.Adam(local_discriminator.parameters(), lr=0.0005)

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
        domain_labels_source = torch.zeros(source_features.size(0), 1).float()  # 源域标签为0
        domain_labels_target = torch.ones(target_features.size(0), 1).float()  # 目标域标签为1
        
        # 反转梯度
        source_features_reversed = GradientReversalLayer.apply(source_features)
        target_features_reversed = GradientReversalLayer.apply(target_features)
        # print("lalalahjh",source_features_reversed.size())
        # print("lalala",target_features_reversed.size())
        
        # 计算鉴别器的输出
        global_output_source = global_discriminator(source_features_reversed)
        global_output_target = global_discriminator(target_features_reversed)

        # domain_labels_source = domain_labels_source.unsqueeze(1)
        # domain_labels_target = domain_labels_target.unsqueeze(1)

        # 计算损失
        global_loss_source = criterion(global_output_source, domain_labels_source)
        global_loss_target = criterion(global_output_target, domain_labels_target)

        global_loss = global_loss_source + global_loss_target
        optimizer_global.zero_grad()
        global_loss.backward(retain_graph=True)
        optimizer_global.step()

        # 提取源域和目标域的预测和特征
        # 如果 source_features 缺少时间维度，添加它
        if source_features.dim() == 2:  # (batch_size, input_size)
            source_features = source_features.unsqueeze(2)  # (batch_size, input_size, 1)

        # 类似处理 target_features
        if target_features.dim() == 2:
            target_features = target_features.unsqueeze(2)

        source_preds = classifier(source_features)  # (batch_size, num_classes)
        target_preds = classifier(target_features)  # (batch_size, num_classes)

        # 结合特征和类别预测进行局部鉴别
        combined_source = local_discriminator(source_features, source_preds)
        combined_target = local_discriminator(target_features, target_preds)

        # 这里开始扩展标签以匹配局部鉴别器输出的维度
        domain_labels_source = torch.zeros(source_features.size(0), 1).float()  # 源域标签
        domain_labels_target = torch.ones(target_features.size(0), 1).float()  # 目标域标签

        # 扩展标签，使其与局部鉴别器输出大小匹配
        domain_labels_source = domain_labels_source.expand_as(combined_source)
        domain_labels_target = domain_labels_target.expand_as(combined_target)

        # 计算局部鉴别器的损失，使用二元交叉熵损失
        local_loss_source = criterion(combined_source, domain_labels_source.expand_as(combined_source))
        local_loss_target = criterion(combined_target, domain_labels_target.expand_as(combined_target))

        # 合并局部损失
        local_loss = local_loss_source + local_loss_target
        optimizer_local.zero_grad()
        local_loss.backward()
        optimizer_local.step()

    print(f'Epoch {epoch+1}, Global Loss: {global_loss.item()}, Local Loss: {local_loss.item()}')
