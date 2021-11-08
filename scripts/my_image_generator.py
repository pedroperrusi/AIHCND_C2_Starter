from tensorflow.keras.preprocessing.image import ImageDataGenerator

import scripts.preprocessing as preprocessing


class MyImageDataGenerator(ImageDataGenerator):
    def __init__(self, **kwargs):
        """ Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        """
        super().__init__(
            preprocessing_function=preprocessing.preprocess_image,
            **kwargs)
