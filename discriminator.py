import torch
import torch.nn as nn
import torch.optim as optim

# 全局鉴别器，D-Conv (1 x 61, 2)
class GlobalDiscriminator(nn.Module):
    def __init__(self, input_size):
        super(GlobalDiscriminator, self).__init__()
        # 定义一个卷积层，输入大小为 1x61，输出为2（源域/目标域的二分类任务）
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=2, kernel_size=61)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.softmax(x)
        return x

# 局部鉴别器，针对每个类别有一个独立的类内鉴别器，D-Conv (1 x 61, 2)
class LocalDiscriminator(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LocalDiscriminator, self).__init__()
        # 为每个类别创建一个鉴别器
        self.local_discriminators = nn.ModuleList([nn.Conv1d(in_channels=input_size, out_channels=2, kernel_size=61) for _ in range(num_classes)])
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, class_idx):
        # 针对给定类别，选择对应的类内鉴别器进行前向传播
        x = self.local_discriminators[class_idx](x)
        x = self.softmax(x)
        return x

# 定义损失函数
loss_fn = nn.BCELoss()  # 使用二分类交叉熵损失

# 示例的训练循环，全局鉴别器
def train_global_discriminator(global_discriminator, feature_extractor, source_loader, target_loader, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        global_discriminator.train()
        running_loss = 0.0
        
        # 源域的数据标签为0
        for inputs, _ in source_loader:
            optimizer.zero_grad()
            inputs = inputs.permute(0, 2, 1)  # 转换输入维度 (batch_size, 通道, 时间点)
            source_features = feature_extractor(inputs)  # 提取源域的特征
            
            domain_labels = torch.zeros(source_features.size(0), 2)  # 源域标签设为0
            domain_labels[:, 0] = 1  # 源域标签设为 [1, 0]
            domain_labels = domain_labels.to(inputs.device)

            outputs = global_discriminator(source_features)
            loss = loss_fn(outputs, domain_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        # 目标域的数据标签为1
        for inputs, _ in target_loader:
            optimizer.zero_grad()
            inputs = inputs.permute(0, 2, 1)  # 转换输入维度
            target_features = feature_extractor(inputs)  # 提取目标域特征

            domain_labels = torch.zeros(target_features.size(0), 2)  # 目标域标签设为1
            domain_labels[:, 1] = 1  # 目标域标签为 [0, 1]
            domain_labels = domain_labels.to(inputs.device)

            outputs = global_discriminator(target_features)
            loss = loss_fn(outputs, domain_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}")

# 示例的局部鉴别器训练
def train_local_discriminator(local_discriminator, classifier, feature_extractor, source_loader, target_loader, optimizer, num_classes, num_epochs=25):
    for epoch in range(num_epochs):
        local_discriminator.train()
        running_loss = 0.0
        
        # 针对每个类别分别计算损失
        for class_idx in range(num_classes):
            # 源域的数据
            for inputs, labels in source_loader:
                optimizer.zero_grad()
                inputs = inputs.permute(0, 2, 1)  # 转换输入维度
                source_features = feature_extractor(inputs)
                class_probs = classifier(inputs)  # 获取分类器的预测概率
                
                # 针对每个类别计算局部鉴别损失
                class_preds = class_probs[:, class_idx].unsqueeze(1)  # 提取当前类别的概率
                domain_labels = torch.zeros(source_features.size(0), 2)  # 源域标签为0
                domain_labels[:, 0] = 1  # 源域标签 [1, 0]
                domain_labels = domain_labels.to(inputs.device)

                outputs = local_discriminator(source_features * class_preds, class_idx)
                loss = loss_fn(outputs, domain_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            # 目标域的数据
            for inputs, _ in target_loader:
                optimizer.zero_grad()
                inputs = inputs.permute(0, 2, 1)
                target_features = feature_extractor(inputs)
                class_probs = classifier(inputs)  # 获取分类器的预测概率
                
                class_preds = class_probs[:, class_idx].unsqueeze(1)
                domain_labels = torch.zeros(target_features.size(0), 2)  # 目标域标签为1
                domain_labels[:, 1] = 1  # 目标域标签为 [0, 1]
                domain_labels = domain_labels.to(inputs.device)

                outputs = local_discriminator(target_features * class_preds, class_idx)
                loss = loss_fn(outputs, domain_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
        
        print(f"Epoch {epoch + 1}/{num_epochs}, Local Discriminator Loss: {running_loss:.4f}")

# 初始化全局和局部鉴别器
input_size = 61  # 根据实际特征的维度
num_classes = 2  # 假设二分类任务
global_discriminator = GlobalDiscriminator(input_size)
local_discriminator = LocalDiscriminator(input_size, num_classes)

# 定义优化器
optimizer_global = optim.Adam(global_discriminator.parameters(), lr=0.001)
optimizer_local = optim.Adam(local_discriminator.parameters(), lr=0.001)

# 你需要有源域和目标域的 DataLoader（source_loader 和 target_loader）
# 以及已经训练好的分类器和特征提取器

# 训练全局和局部鉴别器
train_global_discriminator(global_discriminator, feature_extractor, source_loader, target_loader, optimizer_global, num_epochs=25)
train_local_discriminator(local_discriminator, classifier, feature_extractor, source_loader, target_loader, optimizer_local, num_classes, num_epochs=25)