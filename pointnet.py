import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#TNet网络，用于生成一个变换矩阵(b,3,3)  3是每个点的维度 也可能是6
class TNet(nn.Module):
    def __init__(self,dim = 3):  #dim为3或6
        super(TNet, self).__init__()
        # 定义一个小型的MLP网络来预测变换矩阵
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, dim*dim)
        self.dim = dim

    def forward(self, x):

        x = F.relu(self.fc1(x))#(b,n,64)
        x = F.relu(self.fc2(x))#(b,n,128)
        x = F.relu(self.fc3(x))#(b,n,1024)

        x = torch.max(x, 1)[0]  # (b,1,1024)
        x = F.relu(self.fc4(x))#(b,1,512)
        x = F.relu(self.fc5(x))#(b,1,256)
        x = self.fc6(x)
        x = x.view(-1,self.dim*self.dim)  ##(b,dim**2)
        # print(x.size())
        iden = torch.eye(self.dim, dtype=torch.float32).view(1, -1).repeat(x.size(0), 1)  #(b,dim**2)
        # print(iden.size())
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1,self.dim,self.dim)  #(32,3,3)
        # print(x.size())
        return x


class PointNet(nn.Module):
    def __init__(self, num_classes=10,dim = 3):
        super(PointNet, self).__init__()
        self.tnet = TNet(dim)  # 输入的变换网络

        # 对每个点的特征进行处理的MLP
        self.fc1 = nn.Linear(dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)

        # 分类层
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, num_classes)
        self.dim = dim

    def forward(self, x):
        # 输入的x为(batch_size, num_points, 3)

        # TNet变换：对点云进行变换
        transform = self.tnet(x)  # (batch_size, 3, 3)
        x = torch.bmm(x, transform)  # 对每个点云批次应用变换 #（b,n,3）*（b,3,3）=（b,n,3）

        # 对每个点的特征进行处理
        x = F.relu(self.fc1(x)) # (batch_size, num_points, 64)
        x = F.relu(self.fc2(x))  # (batch_size, num_points, 128)
        x = F.relu(self.fc3(x))  # (batch_size, num_points, 1024)

        # 最大池化层：从每个点的特征中提取全局特征
        x = torch.max(x, 1)[0]  # (batch_size, 1024)

        # 分类网络
        x = F.relu(self.fc4(x)) # (batch_size, 512)
        x = F.relu(self.fc5(x))  # (batch_size, 256)
        x = self.fc6(x)  # (batch_size, num_classes)

        return x


# 测试PointNet模型
if __name__ == "__main__":
    # 假设输入的点云数据形状为 (batch_size, num_points, 3)
    batch_size = 4
    num_points = 500
    point_cloud = torch.randn(batch_size, num_points, 6)  # 随机生成模拟点云数据
    # model = TNet()
    model = PointNet(num_classes=10,dim=6)  #dim可以是3也可以是6  6就是加上了三个法向量
    output = model(point_cloud)
    print("Output shape:", output.shape)  # 输出应该是 (batch_size, num_classes)
    predict = torch.argmax(output,dim = 1)
    print(predict.shape)
