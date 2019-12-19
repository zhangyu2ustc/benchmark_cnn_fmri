#!/home/yuzhang/tensorflow-py3.6/bin/python3.6

# Author: Yu Zhang
# License: simplified BSD
# coding: utf-8

###fMRI decoding: using event signals instead of activation pattern from glm

import numpy as np

from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, LearningRateScheduler, Callback
from tensorflow.keras.utils import to_categorical
##from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.layers import Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv3D, MaxPooling3D, BatchNormalization, AveragePooling3D, GlobalAveragePooling3D, Add
from tensorflow.keras.models import Model
import tensorflow.keras as keras

import tensorflow.keras.backend as K
import tensorflow as tf
K.set_image_data_format('channels_last')

global img_resize, Flag_CNN_Model, Flag_Simple_RESNET, Nlabels
from configure_fmri import *

#####build different neural networks
####################################################################
def bn(x, bn_axis=-1, zero_init=False):
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    elif K.image_data_format() == 'channels_last':
        bn_axis = -1
    return BatchNormalization(axis=bn_axis, fused=True,momentum=0.9, epsilon=1e-5,gamma_initializer='zeros' if zero_init else 'ones')(x)

def conv2d(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

def conv3d(x, filters, kernel, strides=1, name=None):
    return Conv3D(filters, kernel,strides=strides, use_bias=False, padding='same',
                  kernel_initializer="he_normal", kernel_regularizer=keras.regularizers.l2(l2_reg))(x)

def conv(x, filters, kernel, flag_2d3d='2d', strides=1):
    if flag_2d3d == '2d':
        return conv2d(x, filters, kernel, strides=strides)
    elif flag_2d3d == '3d':
        return conv3d(x, filters, kernel, strides=strides)

def identity_block(input_tensor, filters, kernel_size, flag_2d3d='2d'):
    filters1, filters2, filters3 = filters

    x = conv(input_tensor, filters1, 1, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters3, 1, flag_2d3d)
    x = bn(x, zero_init=True)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, filters, kernel_size, strides=2,flag_2d3d='2d'):
    filters1, filters2, filters3 = filters

    x = conv(input_tensor, filters1, 1, flag_2d3d)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, flag_2d3d, strides=strides)
    x = bn(x)
    x = Activation('relu')(x)

    x = conv(x, filters3, 1, flag_2d3d)
    x = bn(x, zero_init=True)

    shortcut = conv(input_tensor,filters3, 1, flag_2d3d, strides=strides)
    shortcut = bn(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_model_resnet50(input_shape, Nlabels=6, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256,flag_print=True):
    #############build resnet50 of 2d and 3d conv
    if not Nlabels:
        global target_name
        Nlabels = len(np.unique(target_name))+1

    global Flag_CNN_Model, Flag_Simple_RESNET

    input0 = Input(shape=input_shape)

    x = conv(input0, filters, convsize2, Flag_CNN_Model, strides=2)
    x = bn(x)
    x = Activation('relu')(x)
    try:
        img_resize   ## using either maxpooling or img_resampling
    except:
        if Flag_CNN_Model == '2d':
            x = MaxPooling2D(poolsize, padding='same')(x)
        elif Flag_CNN_Model == '3d':
            x = MaxPooling3D(poolsize, padding='same')(x)

    x = conv_block(x, [filters, filters, filters*2], convsize, strides=1, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2], convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2], convsize, flag_2d3d=Flag_CNN_Model)

    filters *= 2
    x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)

    if not Flag_Simple_RESNET:
        filters *= 2
        x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
        x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)


    filters *= 2
    x = conv_block(x, [filters, filters, filters*2], convsize, strides=2, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)
    x = identity_block(x, [filters, filters, filters*2],convsize, flag_2d3d=Flag_CNN_Model)

    if Flag_CNN_Model == '2d':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif Flag_CNN_Model == '3d':
        x = GlobalAveragePooling3D(name='avg_pool')(x)

    out = Dense(Nlabels, activation='softmax')(x)

    model = tf.keras.models.Model(input0, out)
    if flag_print:
        model.summary()

    return model

####change: reduce memory but increase parameters to train
def build_cnn_model_test(input_shape, Nlabels=6, filters=16, convsize=3, convsize2=5, poolsize=2, hidden_size=256, conv_layers=4,flag_print=True):
    #     import keras.backend as K
    #     if K.image_data_format() == 'channels_first':
    #         img_shape = (1,img_rows,img_cols)
    #     elif K.image_data_format() == 'channels_last':
    #         img_shape = (img_rows,img_cols,1)

    if not Nlabels:
        global target_name
        Nlabels = len(np.unique(target_name)) + 1

    global Flag_CNN_Model

    input_tensor = Input(shape=input_shape)    ##use tensor instead of shape
    ####quickly reducing image dimension first
    x = conv(input_tensor, filters, convsize2, Flag_CNN_Model, strides=2)
    x = bn(x)
    x = Activation('relu')(x)

    for li in range(conv_layers-1):
        x = conv(x, filters, convsize, Flag_CNN_Model)
        x = bn(x)
        x = Activation('relu')(x)

        x = conv(x, filters, convsize, Flag_CNN_Model)
        x = bn(x)
        x = Activation('relu')(x)

        x = conv(x, filters*2, convsize, Flag_CNN_Model, strides=2)
        x = bn(x)
        x = Activation('relu')(x)

        x = Dropout(0.25)(x)
        filters *= 2
        #if (li+1) % 2 == 0:
        #    filters *= 2

    if Flag_CNN_Model == '2d':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
    elif Flag_CNN_Model == '3d':
        x = GlobalAveragePooling3D(name='avg_pool')(x)

    out = Dense(Nlabels, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=out)
    if flag_print:
        model.summary()

    '''
    adam = Adam(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    '''
    return model
