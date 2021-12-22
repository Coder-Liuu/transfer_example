import paddle
from paddle.vision.models import mobilenet_v2
from paddle.vision.datasets import DatasetFolder
import paddle.vision.transforms as T
from paddle.static import InputSpec
import paddle.nn as nn


# 数据集准备
data_transforms = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[127.5, 127.5, 127.5],
        std=[127.5, 127.5, 127.5],
    )
])
# 加载数据集
train_dir = 'data/tiny_image/train/'
train_dataset = DatasetFolder(train_dir, transform=data_transforms)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=32)

val_dir = 'data/tiny_image/val/'
val_dataset = DatasetFolder(val_dir, transform=data_transforms)
val_loader = paddle.io.DataLoader(val_dataset, batch_size=32)

# 定义模型
model = mobilenet_v2(pretrained=True)
model.classifier = nn.Sequential(
    nn.Linear(model.last_channel, 256),
    nn.ReLU(),
    nn.Linear(256, 2),
    nn.Softmax()
)

# 冻结参数
for param in model.features.parameters():
    param.stop_gradient = True

# 模型初始化
input_ = InputSpec([None, 3, 224, 224], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(model, input_, label)
print(model.summary())

# 模型训练
optim = paddle.optimizer.Adam(parameters=model.parameters())
print("start train")
model.prepare(optim, nn.CrossEntropyLoss(), paddle.metric.Accuracy())
model.fit(train_loader,val_loader,epochs=1)
