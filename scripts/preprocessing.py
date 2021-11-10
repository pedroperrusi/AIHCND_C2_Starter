"""
Handy functions for image pre-processing
"""

import sklearn.preprocessing

from cv2 import cv2
import skimage
import numpy as np


def normalize_image(image):
    return (image - np.mean(image)) / np.std(image)


def min_max_scaling(image, feature_range=(-1, 1)):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=feature_range)
    scaler.fit(image)
    return scaler.transform(image)


def clahe_scaling(image, clipLimit=3.0, tileGridSize=(16, 16)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    return clahe.apply(image)


def preprocess_image(src):
    """Takes an input image and returns a modified version of it"""
    IMG_SIZE = (224, 224)
    INPUT_SHAPE = (1, 224, 224, 3)
    dst = cv2.resize(src, IMG_SIZE) / 255
    if src.shape[2] == 3:
        dst = cv2.cvtColor(dst, cv2.COLOR_RGB2GRAY)
    dst = skimage.img_as_ubyte(dst)
    dst = clahe_scaling(dst)
    dst = skimage.img_as_float(dst)
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_GRAY2RGB) * 255
    return np.broadcast_to(dst, INPUT_SHAPE)
