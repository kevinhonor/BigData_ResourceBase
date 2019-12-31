import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]#文件名称

classes = ["cp"]#类别


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))#每一个文件的图片xml
    tree=ET.parse(in_file)
    root = tree.getroot()#解析xml
    for obj in root.iter('object'):#取其中的对象标签
        difficult = obj.find('difficult').text#不随和的（就是不行的）
        cls = obj.find('name').text#取里面的名称
        if cls not in classes or int(difficult)==1:#如果不随和就跳过
            continue
        cls_id = classes.index(cls)#在类型索引的位置
        print(cls_id)
        xmlbox = obj.find('bndbox')#图片框
        #print(image_id)
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        #取里面便宜和最大小之
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
#现在在的绝对路径
wd = getcwd()

for year, image_set in sets:
    #打开文件
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))#写入文件的完整路径
        convert_annotation(year, image_id, list_file)#年，图片名，文件名称
        list_file.write('\n')
    list_file.close()
