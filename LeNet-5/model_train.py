import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from cnn_model import CNN
import os

# 超参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001

# 加载 MNIST 数据集
train_data = torchvision.datasets.MNIST(
    root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=True
)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_x = torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:2000] / 255.
test_y = test_data.targets[:2000]

# 定义模型
cnn = CNN(num_classes=10)
optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

# 训练和测试
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = (pred_y == test_y.numpy()).mean()
            print(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

if 'model' not in os.listdir('./'):
    os.mkdir('model')
torch.save(cnn, 'zjj.pkl')
