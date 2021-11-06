import tensorflow as tf


class VGG16:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.name = 'VGG16'

    def get_preprocessing(self):
        return tf.keras.models.Sequential(tf.keras.layers.Lambda(tf.keras.applications.vgg16.preprocess_input,
                                                                 input_shape=self.input_shape),
                                          name=self.name + '_preprocessing')

    def get_base_model(self):
        base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_shape=self.input_shape)
        return base_model

    def get_fine_tuning_top_layer(self):
        """ Fine tune last two convolution blocks """
        return 8


class ResNet50:
    def __init__(self, input_shape=(224, 224, 3)):
        self.input_shape = input_shape
        self.name = 'ResNet50'

    def get_preprocessing(self):
        return tf.keras.models.Sequential(tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                                                                 input_shape=self.input_shape),
                                          name=self.name + '_preprocessing')

    def get_base_model(self):
        base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=self.input_shape)
        return base_model

    def get_fine_tuning_top_layer(self):
        """ Fine tune last convolution blocks """
        return 30
