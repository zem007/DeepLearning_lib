import numpy as np
import random
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

def keras_image_generator_2D():
    """ Keras image generator 2d
        Args:
            None
        Returns:
            A generator instance
    """
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

def image_generator_2Dto3D_with_mask(data, label, input_generator, class_num, aug_planes = ['xy', 'yz', 'xz'], batch_size = 1, shuffle=True, seed = 0):
    """ a 3d generator wraper based on 2d keras image generator
        Args:
            data: np.array of shape (n, width, height, channel, 1)
            label: np.array of shape (n, width, height, channel)
            input_generator: a generator instance
            class_num: int, how many classes to be classified
            aug_planes: list of str, indicate what planes should be augmented
            batch_size: int
            shuffle: bool
        Returns:
            a generator wraper. can be used by (x, y) = next(gen)
    """
    data = np.squeeze(data, axis = -1)  #channel last
    seed = seed
    xy_data = data
    xy_label = label
    xy_data_generator = input_generator.flow(xy_data, batch_size=batch_size, shuffle = shuffle, seed = seed)
    xy_label_generator = input_generator.flow(xy_label, batch_size=batch_size, shuffle = shuffle, seed = seed)
    zy_data = data.transpose(0,3,2,1)
    zy_label = label.transpose(0,3,2,1)
    zy_data_generator = input_generator.flow(zy_data, batch_size=batch_size, shuffle = shuffle, seed = seed)
    zy_label_generator = input_generator.flow(zy_label, batch_size=batch_size, shuffle = shuffle, seed = seed)
    xz_data = data.transpose(0,1,3,2)
    xz_label = label.transpose(0,1,3,2)
    xz_data_generator = input_generator.flow(xz_data, batch_size=batch_size, shuffle = shuffle, seed = seed)
    xz_label_generator = input_generator.flow(xz_label, batch_size=batch_size, shuffle = shuffle, seed = seed)
    while True:
        plane = random.choice(aug_planes)
        if plane == 'xy' or plane == 'yx':
            data_batch = next(xy_data_generator)
            label_batch = next(xy_label_generator)
            yield(np.expand_dims(data_batch, axis=-1), np_utils.to_categorical(label_batch, class_num))
            
        elif plane == 'yz' or plane == 'zy':
            data_batch = next(zy_data_generator)
            label_batch = next(zy_label_generator)
            data_batch = data_batch.transpose(0,3,2,1)
            label_batch = label_batch.transpose(0,3,2,1)
            yield(np.expand_dims(data_batch, axis=-1), np_utils.to_categorical(label_batch, class_num))
    
        elif plane == 'xz' or plane == 'zx':
            data_batch = next(xz_data_generator)
            label_batch = next(xz_label_generator)
            data_batch = data_batch.transpose(0,1,3,2)
            label_batch = label_batch.transpose(0,1,3,2)
            yield(np.expand_dims(data_batch, axis=-1), np_utils.to_categorical(label_batch, class_num))
