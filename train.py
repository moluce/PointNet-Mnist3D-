import torch
from tqdm import tqdm
from pointnet import  PointNet


# 加载数据 训练集和测试集
train_dataloader = torch.load('train_point_cloud_batches.pt')
test_dataloader = torch.load('test_point_cloud_batches.pt')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = PointNet(10,dim=6)
model.to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(),lr=0.001)
epochs = 20

for epoch in range(epochs):

    #训练
    model.train()
    print(f"开始第{epoch}轮训练")
    all_loss = 0
    for points, labels in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        points = points.to(device)
        labels = labels.to(device)
        # print(points.shape)
        # print(labels.shape)

        output = model(points)
        optim.zero_grad()
        loss = loss_fn(output,labels)
        loss.backward()
        optim.step()

        all_loss += loss.item()

    print(f'第{epoch}轮，all_loss = {all_loss}')


    # 测试
    model.eval()
    print(f"开始第{epoch}轮测试")
    all_count = 0
    correct_count = 0
    for points, labels in tqdm(test_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
        points = points.to(device)
        labels = labels.to(device)
        # print(points.shape)
        # print(labels.shape)
        output = model(points)
        correct_count += (torch.argmax(output, 1) == labels).sum().item()
        all_count += points.size()[0]
    accuarcy = correct_count/all_count
    print(f'第{epoch}轮，准确率 = {accuarcy}')
    torch.save(model, f'logs/epoch_{epoch}，Accuarcy_{accuarcy}.pth')


