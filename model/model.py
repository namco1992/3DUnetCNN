import numpy as np
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, Highway, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adamax, Adam, Nadam
from keras.layers.advanced_activations import PReLU, LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.merge import concatenate


def conv_block(x_input, num_filters, pool=True, norm=False, drop_rate=0.0):

    x1 = Conv3D(num_filters, (3, 3, 3), border_mode='same', W_regularizer=l2(1e-4))(x_input)
    x1 = PReLU()(x1)
    if norm:
        x1 = BatchNormalization()(x1)
    if drop_rate > 0.0:
        x1 = GaussianDropout(drop_rate)(x1)
    if pool:
        x1 = MaxPooling3D()(x1)
    x_out = x1
    return x_out


def dense_branch(xstart, name, outsize=1, activation='sigmoid'):
    xdense_ = Dense(32, W_regularizer=l2(1e-4))(xstart)
    xdense_ = BatchNormalization()(xdense_)
    # xdense_ = GaussianDropout(0)(xdense_)
    # xdense_ = PReLU()(xdense_)
    xout = Dense(outsize, activation=activation, name=name, W_regularizer=l2(1e-4))(xdense_)
    return xout


def build_model(input_shape):

    xin = Input(input_shape)

    # shift the below down by one
    x1 = conv_block(xin, 8, norm=True, drop_rate=0)  #outputs 9 ch
    x1_ident = AveragePooling3D()(xin)
    x1_merged = concatenate([x1, x1_ident], axis=-1)

    x2_1 = conv_block(x1_merged, 24, norm=True, drop_rate=0)  #outputs 16+9 ch  = 25
    x2_ident = AveragePooling3D()(x1_ident)
    x2_merged = concatenate([x2_1, x2_ident], axis=-1)

    # by branching we reduce the #params
    x3_1 = conv_block(x2_merged, 64, norm=True, drop_rate=0)  #outputs 25 + 16 ch = 41
    x3_ident = AveragePooling3D()(x2_ident)
    x3_merged = concatenate([x3_1, x3_ident], axis=-1)

    x4_1 = conv_block(x3_merged, 72, norm=True, drop_rate=0)  #outputs 25 + 16 ch = 41
    x4_ident = AveragePooling3D()(x3_ident)
    x4_merged = concatenate([x4_1, x4_ident], axis=-1)

    x5_1 = conv_block(x4_merged, 72, norm=True, pool=False, drop_rate=0)  #outputs 25 + 16 ch = 41

    xpool = GlobalMaxPooling3D()(x5_1)
    # xpool_norm = BatchNormalization()(xpool)

    xout_malig = dense_branch(xpool, name='o_mal', outsize=1, activation='sigmoid')

    model = Model(input=xin, output=xout_malig)
    lr_start = 1e-4
    opt = Nadam(lr_start, clipvalue=1.0)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


# def unet_model_3d(input_shape, downsize_filters_factor=1, pool_size=(2, 2, 2), n_labels=1,
#                   initial_learning_rate=0.00001, deconvolution=False):
#     """
#     Builds the 3D UNet Keras model.
#     :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size).
#     :param downsize_filters_factor: Factor to which to reduce the number of filters. Making this value larger will
#     reduce the amount of memory the model will need during training.
#     :param pool_size: Pool size for the max pooling operations.
#     :param n_labels: Number of binary labels that the model is learning.
#     :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
#     :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
#     increases the amount memory required during training.
#     :return: Untrained 3D UNet Model
#     """
#     inputs = Input(input_shape)
#     conv1 = Conv3D(int(32/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(inputs)
#     conv1 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv1)
#     pool1 = MaxPooling3D(pool_size=pool_size)(conv1)

#     conv2 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool1)
#     conv2 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv2)
#     pool2 = MaxPooling3D(pool_size=pool_size)(conv2)

#     conv3 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool2)
#     conv3 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv3)
#     pool3 = MaxPooling3D(pool_size=pool_size)(conv3)

#     conv4 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(pool3)
#     conv4 = Conv3D(int(512/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv4)

#     up5 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=2,
#                      nb_filters=int(512/downsize_filters_factor), image_shape=input_shape[-3:])(conv4)
#     up5 = concatenate([up5, conv3], axis=1)
#     conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up5)
#     conv5 = Conv3D(int(256/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv5)

#     up6 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=1,
#                      nb_filters=int(256/downsize_filters_factor), image_shape=input_shape[-3:])(conv5)
#     up6 = concatenate([up6, conv2], axis=1)
#     conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up6)
#     conv6 = Conv3D(int(128/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv6)

#     up7 = get_upconv(pool_size=pool_size, deconvolution=deconvolution, depth=0,
#                      nb_filters=int(128/downsize_filters_factor), image_shape=input_shape[-3:])(conv6)
#     up7 = concatenate([up7, conv1], axis=1)
#     conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(up7)
#     conv7 = Conv3D(int(64/downsize_filters_factor), (3, 3, 3), activation='relu', padding='same')(conv7)

#     conv8 = Conv3D(n_labels, (1, 1, 1))(conv7)
#     act = Activation('sigmoid')(conv8)
#     model = Model(inputs=inputs, outputs=act)

#     model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coef_loss, metrics=[dice_coef])

#     return model

if __name__ == '__main__':
    build_model((144, 144, 144, 4))
