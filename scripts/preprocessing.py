"""
Handy functions for image pre-processing
"""

import sklearn.preprocessing
import numpy as np
from cv2 import cv2


def normalize_image(image):
    return (image - np.mean(image)) / np.std(image)


def min_max_scaling(image):
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(image)
    return scaler.transform(image)


def clahe_scaling(image, clipLimit=5.0, tileGridSize=(16, 16)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    return clahe.apply(image)
