import os
from PIL import Image
import numpy as np
from keras.utils import to_categorical
from keras.models import load_model

def read_image(img_name):
    im = Image.open(img_name).convert('L')
    data=np.array(im)
    return data

def read_lable(img_name):
    basename = os.path.basename(img_name)
    data = basename.split('_')[0]
    return data
images=[]
lables=[]
for fn in os.listdir('cb'):
    fd=os.path.join('cb',fn)
    images.append(read_image(fd))
    lables.append(int(read_lable(fd)))

y_train=np.array(lables)#转2维数组

y_train=to_categorical(y_train,10)
#对标签进行独热编码处理
x_train=np.array(images)
print(x_train.shape)
x_train=x_train.reshape(10,784)#对数组展平

x_train = x_train / 255.0#2维数据的点缩小到0和1之间


model = load_model('自制手写数据集模型.h5')
predictions = model.predict(x_train)
acc=[]
ac=[]
for j,i in enumerate(predictions):
    acc.append(np.argmax(i))
    ac.append(np.argmax(y_train[j]))
print('正确答案',ac)
print('预测答案',acc)



