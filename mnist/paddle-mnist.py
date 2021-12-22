import numpy as np
import paddle
import paddle.nn as nn
from paddle.metric import Accuracy
from paddle.io import TensorDataset
from paddle.static import InputSpec


# 初始化数据集
data = np.load("data/mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
x_train, x_test = x_train/255.0, x_test/255.0
print("x_train:",x_train.shape, "y_train:",y_train.shape)
print("x_test:",x_train.shape, "y_test:",y_train.shape)

# 转换格式 很重要
x_train = paddle.to_tensor(x_train.astype("float32"))
y_train = paddle.to_tensor(y_train.astype("int64"))
train_dataset = TensorDataset([x_train, y_train])
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64)

x_test = paddle.to_tensor(x_test.astype("float32"))
y_test = paddle.to_tensor(y_test.astype("int64"))
test_dataset = TensorDataset([x_test, y_test])
test_loader = paddle.io.DataLoader(test_dataset, batch_size=64)

# 定义模型
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 256), 
    nn.ReLU(),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
    nn.Softmax()
)
print(model)

# 模型初始化
input_ = InputSpec([None, 1, 28, 28], 'float32', 'image')
label = InputSpec([None, 1], 'int64', 'label')
model = paddle.Model(model,input_,label)
print(model.summary())

# 模型训练
optim = paddle.optimizer.Adam(parameters=model.parameters())
model.prepare(optim, nn.CrossEntropyLoss(), Accuracy())
model.fit(train_loader,test_loader,epochs=5)
