from keras.utils import to_categorical
import numpy as np
import cv2
from PIL import Image
import os
from os.path import basename
from keras.models import load_model
from keras import backend as k

def read_image(img_name):#提取图片
    im = cv2.imread(img_name)#设置为0为灰度图读取，设置为1设置为彩色图片读取
    im1=Image.open(img_name)
    # print(type(im),'opencv')
    # print(type(np.array(im1)), 'pil')
    #print(type(im1),'opencv  --> pil')
    new_img=cv2.resize(im,(300,300),interpolation = cv2.INTER_AREA)
    data=np.array(new_img)#改成2维
    return data

def read_lable(img_name):#取标签
    basenames = basename(img_name)#图片名称
    data = basenames[:3]#对图片名称切片
    data = 0 if data == 'cat' else 1  # 是猫就是0否则1
    return data

def main(path):
    images=[]#创建数组用于存放图片数据
    lables=[]#放标签
    print("读取图片---------------------")
    imgs=os.listdir(path)
    for i in imgs:
        im=os.path.join(path,i)#图片的路径
        #print(np.array(img).shape)
        images.append(read_image(im))
        lables.append(read_lable(im))
    return img_lable(images,lables)

def img_lable(images,lables):
    print('读取完成---------------------')
    print("数据转换中-------------------")
    y_train=np.array(lables)#转2维数组
    x_train=np.array(images)
    x_train = x_train/255.0
    print('标签独热编码处理中------------')
    y_train=to_categorical(y_train,2)
    return model(x_train,y_train)

def model(x_train,y_train):
    model = load_model('图片数据集模型.h5')
    # predictions = model.predict(x_train)#设置取验证集10的图片的数据
    # an=['cat','pds']
    # acc=[]
    # ac=[]
    # for j,i in enumerate(predictions):
    #     acc.append(an[int(np.argmax(i))])
    #     ac.append(an[int(np.argmax(y_train[j]))])
    # print('正确答案',ac)
    # print('预测答案',acc)

if __name__ == '__main__':
   main('test')




