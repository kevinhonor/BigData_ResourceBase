import os#系统模块
import numpy as np
from keras.optimizers import SGD
from os.path import basename
from keras.utils import to_categorical

from keras import  Sequential
from keras.layers import Dense
import cv2

def read_image(img_name):#提取图片
    #opencv读取图片转灰度
    im = cv2.imread(img_name,0)#设置为0为灰度图读取，设置为1设置为彩色图片读取

    #pil读取图片转灰度
    #from PIL import Image
    #im = Image.open(img_name).convert('L')  # 打开图片，('L')转黑白

    data=np.array(im)#改成2维
    return data

def read_lable(img_name):#取标签
    basenames = basename(img_name)#图片名称
    data = basenames[0]#对图片名称切片
    return data

images=[]#创建数组用于存放图片数据
lables=[]#放标签
for fn in os.listdir('ca'):#循环文件中的图片
    fd=os.path.join('ca',fn)#重组图片路径
    images.append(read_image(fd))#用取图片的方法，取每个图片的每个数据点
    lables.append(int(read_lable(fd)))#取标签
print(images)

y_train=np.array(lables)#转2维数组

y_train=to_categorical(y_train,10)#对标签进行独热编码处理

y_test=y_train#用训练数据进行测试

x_train=np.array(images)

x_train=x_train.reshape(x_train.shape[0],-1)#对数组展平
#(6000,28,28)
x_test=x_train
print(type(x_train))
x_train = x_train / 255.0#2维数据的点缩小到0和1之间
x_test = x_test / 255.0#对数据的归一化

# 使用keras的sequential设置网络结构
model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(784, )))#定义输入的形状
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
#model.summary()#查看结构
# 编译模型
model.compile(optimizer=SGD(),#使用梯度下降法，也可以设置其他的方法
              loss='categorical_crossentropy',#使用二次代价函数或交叉熵等方法#损失函数，sparse_categorical_crossentropy交叉熵
              #随机'categorical_crossentropy
              metrics=['accuracy'])#衡量工具
# # 训练模型
# model.fit(x_train, y_train, epochs=200,batch_size=32,validation_data=(x_test,y_test))#传入训练图片数据和标签，还有训练次数和一次性的数量
# # 测试模型
# test_loss, test_acc = model.evaluate(x_test, y_test)#降测试的数据穿入
# print("loss:" ,test_loss)
# print('Test accuracy:',test_acc)
# # 保存模型文件
# model.save('自制手写数据集模型.h5')

