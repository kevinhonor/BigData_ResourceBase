#导入模块
import os#导入系统模块
import random#导入随机模块

#指定文件的写入xml文件和输出的txt文件
xmlfliepath='目标检测\VOCdevkit\VOC2007\Annotations'#图片的xml文件路径
savefilepath='目标检测\VOCdevkit\VOC2007\ImageSets/Main/'#保存的txt文件路径

#数据集的数据分配
trainval_percent=0.1#验证集百分比
train_percent=0.9#训练集百分比

total_xml=os.listdir(xmlfliepath)#将图片的xml读取放入列表
lens=len(total_xml)#列表的长度(14)
print('lens:',lens)
list=range(lens)#将列表的长度作为索引循环保存

#数据分割的个数
tv=int(trainval_percent*lens)#取验证集10%的数据
tr=int(tv*train_percent)#取测试集9%的数据
#取准确的数据
trainval= random.sample(list,tv)#从range(lens)中随机取tv个数出来
train=random.sample(trainval,tr)#从trainval中取9%的准确数据

ftrainval = open(os.path.join(savefilepath, 'trainval.txt'), 'w')
ftest = open(os.path.join(savefilepath, 'test.txt'), 'w')
ftrain = open(os.path.join(savefilepath, 'train.txt'), 'w')
fval = open(os.path.join(savefilepath, 'val.txt'), 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'#图片名称加\n
    if i in trainval:
        ftrainval.write(name)#分类
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()










