import numpy as np
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# 初始化数据集
data = np.load("data/mnist.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]
x_train, x_test = x_train/255.0, x_test/255.0
print("x_train:",x_train.shape, "y_train:",y_train.shape)
print("x_test:",x_train.shape, "y_test:",y_train.shape)


# 定义模型
model = Sequential([
    Input(shape=(28,28)),
    Flatten(),
    Dense(256,activation='relu'),
    Dense(64,activation='relu'),
    Dense(10,activation='softmax'),
])
print(model.summary())

# 训练模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train,y_train,batch_size=64,epochs=10,validation_data=(x_test,y_test))
