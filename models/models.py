""" Tensorflow """
import tensorflow as tf

""" Keras """
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
# from keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
# import keras


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


def down(filters, input_):
    down_ = Conv2D(filters, (3, 3), padding='same')(input_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_ = Activation('relu')(down_)
    down_ = Conv2D(filters, (3, 3), padding='same')(down_)
    down_ = BatchNormalization(epsilon=1e-4)(down_)
    down_res = Activation('relu')(down_)
    down_pool = MaxPooling2D((2, 2), strides=(2, 2))(down_)
    return down_pool, down_res


def up(filters, input_, down_):
    up_ = UpSampling2D((2, 2))(input_)
    up_ = concatenate([down_, up_], axis=3)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    up_ = Conv2D(filters, (3, 3), padding='same')(up_)
    up_ = BatchNormalization(epsilon=1e-4)(up_)
    up_ = Activation('relu')(up_)
    return up_


def Center(filters, input_):
    center = Conv2D(filters, (3, 3), padding='same')(input_)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    center = Conv2D(filters, (3, 3), padding='same')(center)
    center = BatchNormalization(epsilon=1e-4)(center)
    center = Activation('relu')(center)

    return center


def init_segmentation_model(input_shape=(768, 768, 3)):
    inputs = Input(shape=input_shape)

    down0, down0_res = down(16, inputs)
    down1, down1_res = down(32, down0)
    down2, down2_res = down(64, down1)
    down3, down3_res = down(128, down2)
    down4, down4_res = down(256, down3)
    down5, down5_res = down(512, down4)

    center = Center(768, down5)

    up5 = up(512, center, down5_res)
    up4 = up(256, up5, down4_res)
    up3 = up(128, up4, down3_res)
    up2 = up(64, up3, down2_res)
    up1 = up(32, up2, down1_res)
    up0 = up(16, up1, down0_res)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)

    return Model(inputs=inputs, outputs=outputs)
