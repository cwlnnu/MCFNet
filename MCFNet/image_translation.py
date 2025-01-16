# import tensorflow as tf

from tensorflow.keras import Model, regularizers
from tensorflow.keras.layers import Conv2D, Dropout, Activation, Dense, BatchNormalization, Lambda, UpSampling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Flatten, LayerNormalization, UpSampling2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.activations import relu, sigmoid, tanh
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as Kb
# from keras.layers.merge import concatenate
from keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
import tensorflow as tf


class Encoder(Model):
    def __init__(self, input_chs, name, l2_lambda=1e-6, dtype="float32"):
        super().__init__(name=name, dtype=dtype)
        self.input_chs = input_chs
        self.l2_lambda = l2_lambda
        self.content_filter = [64, 64, 128, 128, 256, 256]

        self.style_Conv2D_1_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_1_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')

        self.style_Conv2D_2_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_2_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

        self.style_Conv2D_3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_3_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_3_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

        self.style_Conv2D_4_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_4_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_4_3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')

        self.style_Conv2D_5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.style_Conv2D_5_3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')

        self.content_Conv2D_1_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_1_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')

        self.content_Conv2D_2_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_2_2 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

        self.content_Conv2D_3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_3_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_3_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

        self.content_Conv2D_4_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_4_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_4_3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')

        self.content_Conv2D_5_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_5_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.content_Conv2D_5_3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')


    def Style_Encoder(self, x, reuse=False, scope='style_encoder'):
        x = self.style_Conv2D_1_1(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_1_2(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        style1 = x

        x = self.style_Conv2D_2_1(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_2_2(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        style2 = x

        x = self.style_Conv2D_3_1(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_3_2(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_3_3(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        style3 = x

        x = self.style_Conv2D_4_1(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_4_2(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_4_3(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        style4 = x

        x = self.style_Conv2D_5_1(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_5_2(x)
        x = relu(x, alpha=0.2)
        x = self.style_Conv2D_5_3(x)
        x = relu(x, alpha=0.2)
        # x = tanh(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        style5 = x

        return style1, style2, style3, style4, style5

    def Content_Encoder(self, x, reuse=False, scope='content_encoder'):
        x = self.content_Conv2D_1_1(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_1_2(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        content1 = x

        x = self.content_Conv2D_2_1(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_2_2(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        content2 = x

        x = self.content_Conv2D_3_1(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_3_2(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_3_3(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        content3 = x

        x = self.content_Conv2D_4_1(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_4_2(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_4_3(x)
        x = relu(x, alpha=0.2)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        content4 = x

        x = self.content_Conv2D_5_1(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_5_2(x)
        x = relu(x, alpha=0.2)
        x = self.content_Conv2D_5_3(x)
        x = relu(x, alpha=0.2)
        # x = tanh(x)
        x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)
        content5 = x

        return content1, content2, content3, content4, content5

    def call(self, inputs, training=False):
        style1, style2, style3, style4, style5 = self.Style_Encoder(inputs)
        content1, content2, content3, content4, content5 = self.Content_Encoder(inputs)

        return (style1, style2, style3, style4, style5), (content1, content2, content3, content4, content5)


class generator(Model):
    def __init__(self, output_chs, name, l2_lambda=1e-6, dtype="float32"):
        super().__init__(name=name, dtype=dtype)
        self.output_chs = output_chs
        self.l2_lambda = l2_lambda

        self.Conv2D_5_1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.Conv2D_5_2 = Conv2D(filters=self.output_chs, kernel_size=3, strides=1, padding='same')

        self.Conv2D_4_1 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.Conv2D_4_2 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')

        self.Conv2D_3_1 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.Conv2D_3_2 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.Conv2D_3_3 = Conv2D(filters=128, kernel_size=3, strides=1, padding='same')

        self.Conv2D_2_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.Conv2D_2_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.Conv2D_2_3 = Conv2D(filters=256, kernel_size=3, strides=1, padding='same')

        self.Conv2D_1_1 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.Conv2D_1_2 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')
        self.Conv2D_1_3 = Conv2D(filters=512, kernel_size=3, strides=1, padding='same')

    def call(self, fusion1, fusion2, fusion3, fusion4, fusion5, training=False):
        trans = UpSampling2D()(fusion5)
        trans = self.Conv2D_1_1(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_1_2(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_1_3(trans)
        trans = relu(trans, alpha=0.2)
        if trans.shape[1] > fusion4.shape[1]:
            trans = trans[:, :-1, :, :]
        if trans.shape[2] > fusion4.shape[2]:
            trans = trans[:, :, :-1, :]
        trans = trans + fusion4

        trans = UpSampling2D()(trans)
        trans = self.Conv2D_2_1(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_2_2(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_2_3(trans)
        trans = relu(trans, alpha=0.2)
        if trans.shape[1] > fusion3.shape[1]:
            trans = trans[:, :-1, :, :]
        if trans.shape[2] > fusion3.shape[2]:
            trans = trans[:, :, :-1, :]
        trans = trans + fusion3

        trans = UpSampling2D()(trans)
        trans = self.Conv2D_3_1(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_3_2(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_3_3(trans)
        trans = relu(trans, alpha=0.2)
        if trans.shape[1] > fusion2.shape[1]:
            trans = trans[:, :-1, :, :]
        if trans.shape[2] > fusion2.shape[2]:
            trans = trans[:, :, :-1, :]
        trans = trans + fusion2

        trans = UpSampling2D()(trans)
        trans = self.Conv2D_4_1(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_4_2(trans)
        trans = relu(trans, alpha=0.2)
        if trans.shape[1] > fusion1.shape[1]:
            trans = trans[:, :-1, :, :]
        if trans.shape[2] > fusion1.shape[2]:
            trans = trans[:, :, :-1, :]
        trans = trans + fusion1

        trans = UpSampling2D()(trans)
        trans = self.Conv2D_5_1(trans)
        trans = relu(trans, alpha=0.2)
        trans = self.Conv2D_5_2(trans)
        # trans = relu(trans, alpha=0.2)
        trans = tanh(trans)
        return trans
