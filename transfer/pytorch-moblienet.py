import torchvision.models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from torch import nn
import torch

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# 加载数据集
train_dir = 'data/tiny_image/train/'
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
train_loader = data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dir = 'data/tiny_image/val/'
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms)
test_loader = data.DataLoader(test_dataset, batch_size=32, shuffle=True)

# 加载预训练模型
model = torchvision.models.mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
    nn.Softmax()
)
print(model)
# 冻结参数
for param in model.features.parameters():
    param.requires_grad = False

cost = nn.CrossEntropyLoss()
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

        _, predict = torch.max(predict, 1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = (predict == y).sum().item()
        print("{}/{} Loss:{}".format(i, counts, loss.item()),correct/32)

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
