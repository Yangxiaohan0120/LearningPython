import os
from keras import datasets, models
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import matplotlib.pyplot as plt


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel('train')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 1.导入数据集并进行初始化
data_path = os.getcwd() + '/mnist.npz'
print(data_path)
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))


# 像素值映射到 0 - 1 之间
train_images, test_images = train_images / 255.0, test_images / 255.0


# 2.初始化模型
model = models.Sequential()

# 3.第一层卷积
model.add(
    Conv2D(filters=64, kernel_size=3, activation="relu", strides=(1, 1), padding='valid', input_shape=(28, 28, 1)))

# 4.第一层池化
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

# 5.第二层卷积
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", strides=(1, 1), padding='valid'))


# 6.第二层池化 # 并Dropout
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
# model.add(Dropout(0.25))

# 7.将矩阵摊平成向量
model.add(Flatten())

# 8.全连接层
model.add(Dense(128, activation='softmax'))

# 9.显示网格结构
model.summary()

# 10.模型编译
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 11.模型训练
train_history = model.fit(train_images, train_labels, epochs=5, validation_split=0.2, batch_size=300)


show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# 12.模型保存
model.save('my_mnist.npz_recognize_number_model.h5')

# 13.模型测试
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(test_labels)))

