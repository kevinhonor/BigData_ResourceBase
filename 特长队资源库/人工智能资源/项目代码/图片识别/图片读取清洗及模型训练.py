from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
import numpy as np
import os
from PIL import Image
from os.path import basename
from keras import regularizers

def read_image(img_name):  # 提取图片
    im = Image.open(img_name)  # 设置为0为灰度图读取，设置为1设置为彩色图片读取
    h=int(np.array(im).shape[0])
    w=int(np.array(im).shape[1])
    if h>w:#如果高比宽长
        new_image = Image.new('RGB', (h, h), (128, 128, 128))  # 创建一个新图
        dx=int((h-w)/2)
        new_image.paste(im,(dx,0))
    else :
        new_image = Image.new('RGB', (w, w), (128, 128, 128))  # 创建一个新图
        dx = int((w -h) / 2)
        new_image.paste(im,(0, dx))
    image = new_image.resize((416, 416), Image.BICUBIC)
    return image

classes=['cat','pds']
def read_lable(img_name):  # 取标签
    basenames = basename(img_name)  # 图片名称
    data = basenames[:3]  # 对图片名称切片
    if data in classes:
        class_data=classes.index(data)
        return class_data

def Transpose_data(img_data):
    image = img_data.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def main(path):
    images = []  # 创建数组用于存放图片数据
    lables = []  # 放标签
    print("读取图片---------------------")
    imgs = os.listdir(path)
    for i in imgs:
        im = os.path.join(path, i)  # 图片的路径
        images.append(np.array(read_image(im)))  # 增强图片，为长宽不同的图片增加灰边
        lables.append(read_lable(im))  # 追加数据

        images.append(np.array(Transpose_data(read_image(im)))) # 追加图片翻转后的数据
        lables.append(read_lable(im))  # 追加数据

    return list_images(images, lables)


def list_images(images,lables):
    y_train = np.array(lables)  # 转2维数组
    x_train = np.array(images)  # 转2维数组
    x_train = x_train / 255.0
    y_train = to_categorical(y_train, 2)
    print(x_train.shape)
    print(y_train.shape)
    x_test = x_train
    y_test = y_train
    return image_lable(x_train, y_train, x_test, y_test)

# 使用keras的sequential设置网络结构
from keras.layers import Dense, Dropout, Conv2D, Activation, MaxPool2D, Flatten
def image_lable(x_train, y_train, *test):
    print("模型构造---------------------")
    model = Sequential()
    model.add(Conv2D(
        filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01),
        input_shape=(416, 416, 3)))
    model.add(Activation('relu'))
    model.add(MaxPool2D((2, 2)))
    model.add(Dropout(0.25))

    model.add(
        Conv2D(filters=32, kernel_size=(3, 3), padding='same', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.001))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    print("模型编译---------------------")
    model.compile(optimizer=SGD(),  # 使用梯度下降法，也可以设置其他的方法
                  loss='categorical_crossentropy',  # 使用二次代价函数或交叉熵等方法#损失函数，sparse_categorical_crossentropy交叉熵
                  metrics=['accuracy'])  # 衡量工具
    print('开始训练模型-----------------')
    #model.fit(x_train, y_train, epochs=16, batch_size=10, validation_data=(test))
    print('保存模型---------------------')
    #model.save('图片数据集模型.h5')
    print('模型训练完成')

if __name__ == '__main__':
    main('train')
