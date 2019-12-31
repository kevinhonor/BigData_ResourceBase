from functools import wraps#修饰符
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate#卷积层
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from functools import reduce

def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

#---------------------------------------------------#
#   darknet53 的主体部分
#---------------------------------------------------#
def darknet_body(image_input):# darknet主体
    #image_input=（416，616，3）
    x = DarknetConv2D_BN_Leaky(32, (3,3))(image_input)#将图片形状输入到特殊卷积块，长，宽不变，通道数调整为32
    #（208，208，64）
    x = resblock_body(x, 64, 1)#经过特殊卷积块输入的特征层，输出的通道数，循环的次数
    #（104，104，128，）
    x = resblock_body(x, 128, 2)#经过上面的卷积层，高宽/2，通道数x2
    #（52，52，256）
    x = resblock_body(x, 256, 8)#x=形状，8=次数
    feat1 = x#提取第一个特征层，传入yolo的解码预测值处理的网络中
    # （26，26，512）
    x = resblock_body(x, 512, 8)
    feat2 = x#提取第二个特征层，传入yolo的解码预测值处理的网络中
    #（13，13，1024）
    x = resblock_body(x, 1024, 4)
    feat3 = x#提取第三个特征层，传入yolo的解码预测值处理的网络中
    return feat1,feat2,feat3#返回三次的数据，将三个提取出的特征层传入主体部分


#--------------------------------------------------#
#   单次卷积
#--------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   卷积块
#   DarknetConv2D + BatchNormalization + LeakyReLU
#- --------------------------------------------------#
def resblock_body(x, num_filters, num_blocks):#卷积块
    x = ZeroPadding2D(((1,0),(1,0)))(x)
    #步长为2，长和宽变为原来的2/1
    x = DarknetConv2D_BN_Leaky(num_filters, (3,3), strides=(2,2))(x)#残差网络的输入
    for i in range(num_blocks):#残差网络
        y = DarknetConv2D_BN_Leaky(num_filters//2, (1,1))(x)#1x1卷积，把通道数压缩，
        y = DarknetConv2D_BN_Leaky(num_filters, (3,3))(y)#3x3，把通道数扩展回原来
        x = Add()([x,y])
    return x

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def make_last_layers(x, num_filters, out_filters):
    # 五次卷积
    x = DarknetConv2D_BN_Leaky(num_filters,   (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters,   (1,1))(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    x = DarknetConv2D_BN_Leaky(num_filters,   (1,1))(x)#进行5次卷积后有5个去向，
    #第一个去向
    #第2个去向

    # 将最后的通道数调整为outfilter
    y = DarknetConv2D_BN_Leaky(num_filters*2, (3,3))(x)
    y = DarknetConv2D(out_filters, (1,1))(y)

    return x, y

#---------------------------------------------------#
#   特征层->最后的输出
#---------------------------------------------------#
def yolo_body(image_input, num_anchors, num_classes):
    # 生成darknet53的主干模型
    # （52，52，256）
    # （26，26，512）
    # （13，13，1024）
    feat1,feat2,feat3 = darknet_body(image_input)#获取图片输出的三个特征层

    darknet = Model(image_input, feat3)

    # 第一个特征层
    # y1=(batch_size,13,13,3,85)
    x, y1 = make_last_layers(darknet.output, 512, num_anchors*(num_classes+5))
    #y1是第一个特征层处理完的结果，x是五次卷积后的结果
    #x=Tensor("leaky_re_lu_57/LeakyRelu:0", shape=(?, ?, ?, 512), dtype=float32)
    x = compose(DarknetConv2D_BN_Leaky(256, (1,1)),UpSampling2D(2))(x)#进行上采样，

    x = Concatenate()([x,feat2])#对26x26的特征层进行结合（堆叠）
    # 第二个特征层
    # y2=(batch_size,26,26,3,85)
    x, y2 = make_last_layers(x, 256, num_anchors*(num_classes+5))#进行上采样，

    x = compose(DarknetConv2D_BN_Leaky(128, (1,1)),UpSampling2D(2))(x)

    x = Concatenate()([x,feat1])#对52x52的特征层进行结合（堆叠）
    # 第三个特征层
    # y3=(batch_size,52,52,3,85)
    x, y3 = make_last_layers(x, 128, num_anchors*(num_classes+5))#进行上采样，

    return Model(image_input, [y1,y2,y3])#52，52，256）
    # （26，26，512）
    # （13，13，1024 三个和形状



