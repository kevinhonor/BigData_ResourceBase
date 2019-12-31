"""各种各样的效用函数."""
from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a
#返回浮点16位0到1的数


def get_random_data(annotation_line, input_shape, random=True, max_boxes=20, jitter=0.3, hue=.1, sat=1.5, val=1.5,
                    proc_img=True):
    '''r实时数据增强的随机预处理'''#输入数据annotation_line，输入形状
    line = annotation_line.split()#['G:\\目标检测\\yolo3-keras-master/VOCdevkit/VOC2007/JPEGImages/00.jpg', '209,279,300,308,0']
    image = Image.open(line[0])#取第一个路径
    iw, ih = image.size#print('iw, ih',iw, ih)iw, ih 500 375  iw, ih 400 300图像的高和宽
    h, w = input_shape#h, w 416 416
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    #[[ 32 111 466 254   0][]]将后面的真实框提取出来放入list转numpy

    # 图片拉伸
    import random
    nw=int(random.uniform(350,480))
    nh=int(random.uniform(350,480))
    image = image.resize((nw,nh),Image.BICUBIC)


    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))#创建一个新图的通道，高宽，背景颜色为灰色
    #print(np.array(image).shape)
    new_image.paste(image, (dx, dy))
    #print(np.array(new_image).shape)
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)#0.1
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)#1.5
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    # import matplotlib.pyplot as plt#查看图片
    # plt.imshow(np.array(image))
    # plt.show()
    #print(np.array(image).shape)
    x = rgb_to_hsv(np.array(image) / 255.)#
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1
    #print(np.array(image).shape)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))#创建一个20行5列的数组
    if len(box) > 0:#判断里面是否有物体位置
        np.random.shuffle(box)#打乱物体位置
        box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx#xmin，xmax
        box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
        if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > w] = w
        box[:, 3][box[:, 3] > h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes: box = box[:max_boxes]#如果物体超过20个就取前20个

        box_data[:len(box)] = box#将计算好的先验框存入box_data

    return image_data, box_data




