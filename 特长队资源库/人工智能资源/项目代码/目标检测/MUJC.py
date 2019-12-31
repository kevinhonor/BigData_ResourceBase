from functools import wraps, reduce


def compose(*funcs):  # reduce
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)


def darknet_body(image_input):  # 取特征层最后三个，第一特殊卷积层输入图片形状，后面用卷积块改变通道，输出需要的三个数
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(image_input)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1, feat2, feat3



def DarknetConv2D_BN_Leaky(*args, **kwargs):  # 特殊卷积块，使用use_bias，标准化，激活函数,compose
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    from keras.layers.normalization import BatchNormalization
    from keras.layers import LeakyReLU
    return compose(DarknetConv2D(*args, **no_bias_kwargs),
                   BatchNormalization(),
                   LeakyReLU(alpha=0.1))


from keras.layers import Conv2D


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):  # 单次卷积，使用正则化，填充，修饰器，Conv2D
    from keras.regularizers import l2
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

def resblock_body(x, num_filters, num_blocks):  # 卷积块残差网络，ZeroPadding2D改变长宽，特殊卷积层改变通道，Add
    from keras.layers import ZeroPadding2D, Add
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = DarknetConv2D_BN_Leaky(num_filters // 3, (1, 1))(x)
        y = DarknetConv2D_BN_Leaky(num_filters, (3, 3))(y)
        x = Add()([x, y])
    return x

def make_last_layers(x, num_filters, out_filters):  # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1, 1))(x)

    y = DarknetConv2D_BN_Leaky(num_filters * 2, (3, 3))(x)
    y = DarknetConv2D(out_filters, (1, 1))(y)
    return x, y

def yolo_body(image_input, num_anchors, num_classes):  # 特征层最后的输出，经过五次卷积改变通道，一个进行上采样，一个向上结合
    feat1, feat2, feat3 = darknet_body(image_input)
    from keras.models import Model
    darknet = Model(image_input, feat3)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))
    from keras.layers import UpSampling2D
    x = compose(DarknetConv2D_BN_Leaky(256, (1, 1)), UpSampling2D(2))(x)
    from keras.layers import Concatenate
    x = Concatenate()([x, feat2])
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))
    x = compose(DarknetConv2D_BN_Leaky(128, (1, 1)), UpSampling2D(2))(x)
    x = Concatenate()([x, feat1])
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))
    return Model(image_input, [y1, y2, y3])



import tensorflow as tf
from keras import backend as K


def yolo_head(feats, anchors, num_classes, input_shape):
    num_anchors = len(anchors)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])  # 先验框张量，五维
    grid_shape = K.shape(feats)[1:3]  # 取出其中形状部分
    # 求出grid的xy_wh,组合成完整的网格
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    # 通过平铺一个x维度，n，n维列表
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    # 获取需要反回的四个数值
    grid = K.cast(grid, K.dtype(feats))
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])  # 展平后的五维

    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))  # 前两个值为xy
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))  # wh

    return grid, feats, box_xy, box_wh


def box_iou(b1, b2):  # iou计算图片先验框交集与并集的差

    b1 = K.expand_dims(b1, -2)  # 在倒数第2个位置增加维度
    b1_xy = b1[...:2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou




def yolo_loss(args, anchors, num_classes, ignore_thresh=.5):
    num_layers = len(anchors) // 3  # 得到先验框的个数整除3
    # 将args的值分割出来
    y_true = args[num_layers:]  # y真实值
    yolo_outputs = args[:num_layers]  # 预测的三个特征
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [1, 2, 3]]  # 先验框
    input_shape = (416, 416)  # 输入形状
    # 得到13x13，26，26，52，52网格
    grid_shape = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_layers)]
    loss = 0  # 用于存放最后返回的loss值
    m = K.shape(yolo_outputs[0])[0]  # 取出图片数量
    mf = K.cast(m, K.dtype(yolo_outputs[0]))  # 改变变量的类型

    for l in range(num_layers):
        object_mask = y_true[1][..., 4:5]  # 图片中是否有物体用01表示
        true_class_probs = y_true[1][..., 5:]  # 图片的类

        grid, raw_pred, pred_xy, pred_wh = yolo_head(  # 对输入的预测特征进行解码
            yolo_outputs[1], anchors[anchor_mask[l]], num_classes, input_shape)

        pred_box = K.concatenate([pred_xy, pred_wh])  # xy_wh进行拼接

        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')  # 转bool



        def loop_body(b, ignore_mask):
            true_box = tf.boolean_mask(y_true[1][b, ..., 0:4], object_mask_bool[b, ..., 0])  # 正确的先验框
            iou = box_iou(pred_box[b], true_box)  # 预测和真实进行求面积交集和并集的差
            best_iou = K.max(iou, axis=-1)  #
            ignore_mask = ignore_mask.write(b, K.cast(best_iou < ignore_thresh,
                                                      K.dtype(true_box)))  # 如果iou小于50%，就将图片和先验框写入
            return b + 1, ignore_mask
        def loop_body(b,ignore_mask):
            true_box=tf.boolean_mask(y_true[1][b,...,:4],object_mask_bool[b,...,0])
            iou=box_iou(pred_box[b],true_box)
            iou=K.max(iou,axis=-1)
            ignore_mask=ignore_mask.write(b,K.cast(iou<ignore_thresh),K.dtype(true_box))

        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        ignore_mask = K.expand_dims(ignore_mask, -1)

        raw_true_xy = y_true[1][..., :2] * grid_shape[1][:] - grid
        raw_true_wh = K.log(y_true[1][..., 2:4] / anchors[anchor_mask[1]] * input_shape[::-1])
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))

        box_loss_scale = 2 - y_true[1][..., 2:3] * y_true[l][..., 3:4]
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5], from_logits=True) + (
                1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[4:5], from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)



        xy_loss = K.sum(xy_loss) / mf
        wh_loss = K.sum(wh_loss) / mf
        confidence_loss = K.sum(confidence_loss) / mf
        class_loss = K.sum(class_loss) / mf
        loss = xy_loss + wh_loss + confidence_loss + class_loss
    return loss



from keras.callbacks import TensorBoard

tensorbord = TensorBoard(log_dir='./logs', histogram_freq=1, write_grads=True)
# mode.fit(callable=[tensorbord])


from keras.callbacks import TensorBoard
tensorbord=TensorBoard(log_dir='',histogram_freq=1,write_grads=True)



import cv2
from PIL import Image

# new_img=cv2.resize(im,(300,300),interpolation = cv2.INTER_AREA)
# new_image = Image.new('RGB', (w, w), (128, 128, 128))
# image = new_image.resize((416, 416), Image.BICUBIC)
# image=image.transpose(Image.FLIP_LEFT_RIGHT)
#
# cv2.resize(im,(416,416),interpolation=cv2.INTER_AREA)
# image.resize((416,416),Image.BICUBIC)



x = rgb_to_hsv(np.array(image) / 255.)
x[..., 0] += hue
x[..., 0][x[..., 0] > 1] -= 1
x[..., 0][x[..., 0] < 0] += 1
x[..., 1] *= sat
x[..., 2] *= val
x[x > 1] = 1
x[x < 0] = 0
image_data = hsv_to_rgb(x)

import numpy as np


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape, max_box=20, hue=0.1, jitter=0.3, val=1.5, sat=1.5):
    line = annotation_line.split()  # 读取数据
    image = Image.open(line[0])
    iw, ih = image.size
    w, h = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    if new_ar < 1:
        nh=int(scale*h)
        nw=int(nh*new_ar)
    else:
        nh=int(scale*w)
        nh=int(nh/new_ar)

    dx=int(rand(0,w-nw))
    dy=int(rand(0,h-nh))
    new_image=Image.new('RGB',(w,h),(128,128,128))
    new_image=new_image.paste(image,(dx,dy))
    image=new_image

    flip=rand()<.5
    if flip: image=image.transpose(Image.FLIP_LEFT_RIGHT)

    hue=rand(-hue,hue)
    sat=rand(1,sat) if rand()<.5 else 1/rand(1,sat)
    val=rand(1,val) if rand()<.5 else 1/rand(1,val)
    from matplotlib.colors import rgb_to_hsv,hsv_to_rgb
    x=rgb_to_hsv(np.array(image)/255.)
    x[...,0]+=hue
    x[...,0][x[...,0]>1]-=1
    x[...,0][x[...,0]<0]+=1
    x[...,1]*=sat
    x[...,1]*=val
    x[x>1]=1
    x[x<0]=0
    image_data=hsv_to_rgb(x)

    box_data=np.zeros((max_box,5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:,[0,2]]=box[:,[0,2]]*nw/iw+dx
        box[:,[1,3]]=box[:,[1,3]]*nh/ih+dy
        if flip: box[:[0,2]]=w-box[:,[2,0]]
        box[:,0:2][box[:,0:2]<0]=0


