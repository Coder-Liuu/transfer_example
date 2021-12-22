import numpy as np
import torch
import torch.nn as  nn
import torch.nn.functional as F

# 初始化数据集
data = np.load("data/mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
x_train, x_test = x_train/255.0, x_test/255.0
print("x_train:",x_train.shape, "y_train:",y_train.shape)
print("x_test:",x_train.shape, "y_test:",y_train.shape)

# 转换格式
x_train,y_train = torch.FloatTensor(x_train),torch.LongTensor(y_train)
dataset = torch.utils.data.TensorDataset(x_train,y_train)
train_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 64, shuffle = True)

x_test,y_test = torch.FloatTensor(x_test),torch.LongTensor(y_test)
dataset = torch.utils.data.TensorDataset(x_test,y_test)
test_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size = 64, shuffle = True)

# 定义模型
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(28 * 28, 256), 
            nn.ReLU(True),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Linear(64, 10),
         )

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.layer(x)
        x = F.softmax(x)
        return x

# 定义模型
model = Net()
print(model)
cost = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

# 训练模型
n_epochs = 10
counts = len(train_loader)
for epoch in range(n_epochs):
    print("Epoch: {}/{}".format(epoch, n_epochs))
    model.train()
    for i, data in enumerate(train_loader):
        x, y = data
        predict = model(x)
        loss = cost(predict,y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print("{}/{} Loss:{}".format(i, counts, loss.item()))

    model.eval()
    correct = 0.0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            x, y = data
            predict = model(x)
            _, predict = torch.max(predict, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
        print("Test Acc: ", correct / total * 100)
