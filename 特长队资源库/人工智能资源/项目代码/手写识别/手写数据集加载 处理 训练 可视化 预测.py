from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
from numpy import argmax
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)=mnist.load_data('MNIST_data')

# import matplotlib.pyplot as plt#查看图片
# img=plt.imshow(x_train[0])
# plt.show()

x_train=np.array(x_train)#转2维
x_test=np.array(x_test)

y_train=to_categorical(y_train,10)#对标签进行独热编码处理
y_test=to_categorical(y_test,10)#对标签进行独热编码处理



x_train=x_train.reshape(x_train.shape[0],-1)#对数组展平
x_test=x_test.reshape(x_test.shape[0],-1)#对数据的转换

x_train = x_train / 255.0#2维数据的点缩小到0和1之间
x_test = x_test / 255.0#对数据的归一化

# 使用keras的sequential设置网络结构
model = Sequential()
model.add(Dense(512,activation='relu',input_shape=(784, )))#定义输入的形状
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))
#model.summary()
# 编译模型
model.compile(optimizer=SGD(),#使用梯度下降法，也可以设置其他的方法
              loss='categorical_crossentropy',#使用二次代价函数或交叉熵等方法#损失函数，sparse_categorical_crossentropy交叉熵
              #随机'categorical_crossentropy
              metrics=['accuracy'])#衡量工具


#---------------------------------------------------#
#如果训练好模型，就把训练模型注释，下面是测试模型
#---------------------------------------------------#
#tensorbord 可视化

tbCallBack = TensorBoard(log_dir="./logs", histogram_freq=1,write_grads=True)

# 训练模型
history=model.fit(x_train, y_train, epochs=10,batch_size=128,
                  validation_data=(x_test,y_test),verbose=1,
                    callbacks=[tbCallBack]
                  )#传入训练图片数据和标签，还有训练次数和一次性的数量


#网址输入查看可视化  http://localhost:6006/#scalars

# # 测试模型
# test_loss, test_acc = model.evaluate(x_test, y_test)#降测试的数据传入
# print("loss:" ,test_loss)
# print('Test accuracy:',test_acc)
# # 保存模型文件
# model.save('手写数据集模型.h5')
#
#
#
#
# #模型读取和预测
# model = load_model('手写数据集模型.h5')
# predictions = model.predict(x_test[:10])#设置取验证集前10的图片的数据
#
# acc=[]
# ac=[]
# for j,i in enumerate(predictions):
#     acc.append(argmax(i))
#     ac.append(argmax(y_test[j]))
# print('正确答案',ac)
# print('预测答案',acc)








