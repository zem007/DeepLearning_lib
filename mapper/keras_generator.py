import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

def keras_image_generator():
    datagen = ImageDataGenerator(
            # 在整个数据集上将输入均值置为 0
            featurewise_center=False,
            # 将每个样本均值置为 0
            samplewise_center=False,
            # 将输入除以整个数据集的 std
            featurewise_std_normalization=False,
            # 将每个输入除以其自身 std
            samplewise_std_normalization=False,
            # 应用 ZCA 白化
            zca_whitening=False,
            # ZCA 白化的 epsilon 值
            zca_epsilon=1e-06,
            # 随机图像旋转角度范围 (deg 0 to 180)
            rotation_range=30,
            # 随机水平平移图像
            width_shift_range=0.1,
            # 随机垂直平移图像
            height_shift_range=0.1,
            # 设置随机裁剪范围
            shear_range=0.,
            # 设置随机缩放范围
            zoom_range=0.,
            # 设置随机通道切换范围
            channel_shift_range=0.,
            # 设置输入边界之外的点的数据填充模式
            fill_mode='nearest',
            # 在 fill_mode = "constant" 时使用的值
            cval=0.,
            # 随机翻转图像
            horizontal_flip=True,
            # 随机翻转图像
            vertical_flip=False)
    return datagen

def image_generator_2Dto3D(data, label, datagen, class_num, batch_size = 8, shuffle=True):  #Input dim (n,64,64,64)
    datagen.fit(data)
    for data_batch, label_batch in datagen.flow(data, label, batch_size=batch_size, shuffle = shuffle):
        yield(np.expand_dims(data_batch, axis=-1), np_utils.to_categorical(label_batch, class_num))


