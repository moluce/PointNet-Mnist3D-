import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PointCloudDataset(Dataset):
    def __init__(self, h5_file_path, max_points=34800):
        self.h5_file_path = h5_file_path
        self.max_points = max_points

        # 打开HDF5文件
        with h5py.File(h5_file_path, 'r') as f:
            self.groups = list(f.keys())  # 获取所有组
            self.data = []
            self.labels = []  # 用来存储每个点云的标签

            for group_name in self.groups:
                points = f[f'{group_name}/points'][:]  # 获取点云数据
                normals = f[f'{group_name}/normals'][:]  # 获取法向量数据
                label = f[group_name].attrs["label"]  # 获取标签数据（作为属性）

                # 填充点云到 max_points
                num_points = points.shape[0]
                if num_points < self.max_points:
                    # 如果点云少于max_points，进行填充
                    padding = np.zeros((self.max_points - num_points, 3))
                    points = np.vstack([points, padding])
                    normals = np.vstack([normals, padding])
                elif num_points > self.max_points:
                    # 如果点云多于max_points，截断
                    points = points[:self.max_points]
                    normals = normals[:self.max_points]

                # 合并点云和法向量
                data = np.hstack([points, normals])  # 形状 (N, 6)
                self.data.append(torch.tensor(data, dtype=torch.float32))  # 转换为Tensor
                self.labels.append(torch.tensor(label, dtype=torch.long))  # 标签作为long类型的Tensor

        self.num_samples = len(self.data)  # 数据集中的样本数量

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]  # 返回点云数据和标签

# 训练数据集路径
h5_file_path = 'test_point_clouds.h5'  # 替换为你的文件路径

# 创建数据集和数据加载器
dataset = PointCloudDataset(h5_file_path, max_points=34800)  # max_points为每个点云的最大点数
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # 设置batch_size



# 保存的路径
save_path = 'test_point_cloud_batches.pt'
# 将 DataLoader 中的数据保存到 list 中
all_batches = []
for batch_idx, (points, labels) in enumerate(dataloader):
    # 你可以根据需求添加条件来控制保存多少个batch
    all_batches.append((points, labels))  # 保存每个batch中的points和labels


# 使用 torch.save 将批次数据保存到文件
torch.save(all_batches, save_path)

print(f"数据已保存到 {save_path}")