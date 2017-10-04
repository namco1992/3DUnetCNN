from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Lambda, Dense, Flatten, Reshape, Highway, Activation, Dropout
from keras.layers.convolutional import Conv3D
from keras.layers.pooling import GlobalMaxPooling3D, MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianDropout
from keras.optimizers import Adamax, Adam, Nadam
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from keras.layers.merge import concatenate


def conv_block(x_input, num_filters, pool=True, norm=False, drop_rate=0.0):

    x1 = Conv3D(num_filters, (3, 3, 3), padding='same', kernel_regularizer=l2(1e-4))(x_input)
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
    xdense_ = Dense(32, kernel_regularizer=l2(1e-4))(xstart)
    xdense_ = BatchNormalization()(xdense_)
    # xdense_ = GaussianDropout(0)(xdense_)
    # xdense_ = PReLU()(xdense_)
    xout = Dense(outsize, activation=activation, name=name, kernel_regularizer=l2(1e-4))(xdense_)
    return xout


def build_model(input_shape, initial_learning_rate, weights=None):

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

    model = Model(inputs=xin, outputs=xout_malig)
    opt = Nadam(initial_learning_rate, clipvalue=1.0)

    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    if weights is not None:
        model.load_weights(weights)
    return model


if __name__ == '__main__':
    build_model((144, 144, 144, 4), 1e-4)
