import os
import cv2
name='train'
imgs=os.listdir(name)
total=0
for i in imgs:
    im=os.path.join(name,i)#图片的路径
    #if os.path.basename(im)[-3:]=='jpg' or os.path.basename(im)[-3:]=='jpeg':
    name1 = os.path.basename(im)
    im=cv2.imread(im)
    name1=name1[:3] 
    #new_img = cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join('train',name1+'_'+str(total)+'.jpg'), im)  # 保存到原文件
    total+=1
#
# from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# import numpy as np
# datagen = ImageDataGenerator(
#     rotation_range = 40,     # 随机旋转角度
#     width_shift_range = 0.2, # 随机水平平移
#     height_shift_range = 0.2,# 随机竖直平移
#     rescale = 1/255,         # 数据归一化
#     shear_range = 20,        # 随机错切变换
#     zoom_range = 0.2,        # 随机放大
#     horizontal_flip = True,  # 水平翻转
#     fill_mode = 'nearest',)  # 填充方式
# import os
# for n,j in enumerate(os.listdir('pds')):
#     j1=os.path.join('pds',j)
#     img = load_img(j1)
#     x = img_to_array(img)
#     #print(x.shape)
#     # 打印结果：
#     # (414, 500, 3)
#
#     # 扩展维度
#     x = np.expand_dims(x, 0)
#    # print(x.shape)
#     # 打印结果：
#     # (1, 414, 500, 3)
#
#     # 生成20张图片
#     i = 0
#     for batch in datagen.flow(x):
#         print(batch)
#         i += 1
#         if i==2:
#             break
#     print('finished!')
#
#




