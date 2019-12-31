import xml.etree.ElementTree as ET
from os import getcwd

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["cp"]
# classes = open(r'model_data/voc_classes.txt').read()


def convert_annotation(year, image_id, list_file):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()  # <Element 'annotation' at 0x0000029747AF2318>

    for obj in root.iter('object'): # obj = <Element 'object' at 0x000001F76A4919F8>
        difficult = obj.find('difficult').text  # 0
        cls = obj.find('name').text            # cp

        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)   # 0
        xmlbox = obj.find('bndbox')  # <Element 'bndbox' at 0x000001E69ED11B88>
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

wd = getcwd()

for year, image_set in sets:
    image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()
