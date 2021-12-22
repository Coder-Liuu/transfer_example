from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pathlib
import numpy as np


# 加载数据集
train_dir = 'data/tiny_image/train'
train_dir = pathlib.Path(train_dir)
class_names = np.array([item.name for item in train_dir.glob('*')])

train_dir = 'data/tiny_image/train'
train_dir = pathlib.Path(train_dir)
class_names = np.array([item.name for item in train_dir.glob('*')])

image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = image_generator.flow_from_directory(directory=str(train_dir),
                                                    batch_size=32,
                                                    target_size=(224, 224),
                                                    shuffle=True,
                                                    classes = list(class_names))

base_model = MobileNet(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# 添加全局平均池化层
x = base_model.output
x = GlobalAveragePooling2D()(x)
# 添加一个全连接层
x = Dense(256, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)
# 构建我们需要训练的完整模型
model = Model(inputs=base_model.input, outputs=predictions)
# 锁住所有的卷积层
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
print(model.summary())
model.fit(train_data_gen,epochs=1)
