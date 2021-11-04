from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cv2 import cv2
import skimage

import scripts.preprocessing as preprocessing


def preprocess(src):
    '''Takes an input image and returns a modified version of it'''
    dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY) / 255
    dst = skimage.img_as_ubyte(dst)
    dst = preprocessing.clahe_scaling(dst)
    dst = skimage.img_as_float(dst)
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_GRAY2RGB) * 255
    return dst.reshape(src.shape)


class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        """ Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        """
        super().__init__(
            preprocessing_function=preprocess,
            **kwargs)
