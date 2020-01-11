# -*- coding: utf-8 -*-
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""ResNet50 model for Keras.

Adapted from tf.keras.applications.resnet50.ResNet50().

Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow.keras import backend
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl

from models_le import ConcreteDropout, SpatialConcreteDropout

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def identity_block_base(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    first_conv_2d = layers.Conv2D(
            filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a')
    x = first_conv_2d(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size, use_bias=False,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_base(input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the second conv layer in the block.
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    conv2d_layer = layers.Conv2D(
            filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a')
    x = conv2d_layer(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c')(x)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    shortcut = layers.Conv2D(
            filters3, (1, 1),
            use_bias=False,
            strides=strides,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '1')(
                    input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON,
                                         name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def identity_block_vi_base(input_tensor, kernel_size, filters, stage, block, num_updates):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    divergence_fn = lambda q, p, ignore: (tfd.kl_divergence(q, p)/num_updates)
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tfpl.Convolution2DFlipout(
            filters1, kernel_size=(1, 1), padding='SAME',
            name=conv_name_base + '2a',
            kernel_divergence_fn=divergence_fn
            )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = tfpl.Convolution2DFlipout(
            filters2, kernel_size=kernel_size, padding='SAME',
            activation=None, name=conv_name_base + '2b',
            kernel_divergence_fn=divergence_fn
            )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = tfpl.Convolution2DFlipout(
            filters3, kernel_size=(1, 1), padding='SAME',
            activation=None, name=conv_name_base + '2c',
            kernel_divergence_fn=divergence_fn
            )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block_vi_base(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), num_updates=1):
    """A block that has a conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the second conv layer in the block.
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    divergence_fn = lambda q, p, ignore: (tfd.kl_divergence(q, p)/num_updates)
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    x = tfpl.Convolution2DFlipout(
            filters1, kernel_size=(1, 1), padding='SAME',
            activation=None, name=conv_name_base + '2a',
            kernel_divergence_fn=divergence_fn
            )(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = tfpl.Convolution2DFlipout(
            filters2, kernel_size=kernel_size, strides=strides, padding='SAME',
            activation=None, name=conv_name_base + '2b',
            kernel_divergence_fn=divergence_fn
            )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = tfpl.Convolution2DFlipout(
            filters3, kernel_size=(1, 1), padding='SAME',
            activation=None, name=conv_name_base + '2c',
            kernel_divergence_fn=divergence_fn
            )(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    shortcut = tfpl.Convolution2DFlipout(
            filters3, kernel_size=(1, 1), strides=strides, padding='SAME',
            activation=None, name=conv_name_base + '1',
            kernel_divergence_fn=divergence_fn
            )(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON,
                                         name=bn_name_base + '1')(shortcut)
    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50_base(input_shape, classes):
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block_base(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_base(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_base(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_base(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        classes,
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
        )(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

def ResNet50_dropout(input_shape, classes):
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block_base(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_base(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_base(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_base(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x, training=True)
    x = layers.Dense(
        classes,
        kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
        bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
        )(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

def identity_block_CDrop_base(input_tensor, kernel_size, filters, stage, block, wd=1e-2, dd=2.):
    """The identity block is the block that has no conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    first_conv_2d = SpatialConcreteDropout(
            layers.Conv2D(
                    filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    name=conv_name_base + '2a'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)
    x = first_conv_2d(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = SpatialConcreteDropout(layers.Conv2D(filters2, kernel_size, use_bias=False,
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = SpatialConcreteDropout(layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def conv_block_CDrop_base(input_tensor,
                    kernel_size,
                    filters,
                    stage,
                    block,
                    strides=(2, 2),
                    wd=1e-2, dd=2.):
    """A block that has a conv layer at shortcut.

    Arguments:
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                    middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the second conv layer in the block.
            num_updates: integer, total steps in an epoch (for weighting the loss)
            dropout_rate: float, always-on dropout rate.
            use_variational_layers: boolean, if true train a variational model

    Returns:
            x: Output tensor for the block.

    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    conv2d_layer = SpatialConcreteDropout(layers.Conv2D(
            filters1, (1, 1), use_bias=False, kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '2a'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)
    x = conv2d_layer(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)
    x = SpatialConcreteDropout(layers.Conv2D(filters2, kernel_size, strides=strides, padding='same',
                      use_bias=False, kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2b'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)
    x = SpatialConcreteDropout(layers.Conv2D(filters3, (1, 1), use_bias=False,
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name=conv_name_base + '2c'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)(x)

    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name=bn_name_base + '2c')(x)
    shortcut = SpatialConcreteDropout(layers.Conv2D(
            filters3, (1, 1),
            use_bias=False,
            strides=strides,
            kernel_initializer='he_normal',
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            name=conv_name_base + '1'), 
            weight_regularizer=wd, 
            dropout_regularizer=dd)(input_tensor)
    shortcut = layers.BatchNormalization(axis=bn_axis,
                                         momentum=BATCH_NORM_DECAY,
                                         epsilon=BATCH_NORM_EPSILON,
                                         name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x

def ResNet50_concreteDropout(input_shape, classes, N):
    # initialize weight regularization params
    # N = len(X_train)
    wd = 1e-2 / N
    dd = 2. / N
    
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    conv_block_CDrop = functools.partial(conv_block_CDrop_base, wd=wd, dd=dd)
    identity_block_CDrop = functools.partial(identity_block_CDrop_base, wd=wd, dd=dd)

    # Stage 2
    x = conv_block_CDrop(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_CDrop(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_CDrop(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_CDrop(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_CDrop(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_CDrop(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_CDrop(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_CDrop(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_CDrop(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_CDrop(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_CDrop(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x, training=True)
    x = ConcreteDropout(
            layers.Dense(
                    classes,
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    ), 
            weight_regularizer=wd, 
            dropout_regularizer=dd
            )(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

def ResNet50_llconcreteDropout(input_shape, classes, N):
    # initialize weight regularization params
    # N = len(X_train)
    wd = 1e-2 / N
    dd = 2. / N
    
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block_base(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_base(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_base(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_base(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x, training=True)
    x = ConcreteDropout(
            layers.Dense(
                    classes,
                    kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                    bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY)
                    ), 
            weight_regularizer=wd, 
            dropout_regularizer=dd
            )(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

def ResNet50_vi_flipout(input_shape, classes, num_updates):
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    conv_block_vi = functools.partial(conv_block_vi_base, num_updates=num_updates)
    identity_block_vi = functools.partial(identity_block_vi_base, num_updates=num_updates)

    # Stage 2
    x = conv_block_vi(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_vi(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_vi(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_vi(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_vi(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_vi(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_vi(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_vi(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_vi(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_vi(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_vi(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_vi(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_vi(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_vi(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_vi(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_vi(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = tfpl.DenseFlipout(classes)(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

def posterior_mean_field(kernel_size, bias_size=0, dtype=None):
    """Posterior function for variational layer."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5))
    variable_layer = tfpl.VariableLayer(
            2 * n, dtype=dtype,
            initializer=tfpl.BlockwiseInitializer([
                    tf.keras.initializers.TruncatedNormal(mean=0., stddev=.05, seed=None),
                    tf.keras.initializers.Constant(np.log(np.expm1(1e-5)))],
        sizes=[n, n]))

    def distribution_fn(t):
        scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, n:])
        return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                               reinterpreted_batch_ndims=1)
    distribution_layer = tfpl.DistributionLambda(distribution_fn)
    return tf.keras.Sequential([variable_layer, distribution_layer])

def prior_trainable(kernel_size, bias_size=0, dtype=None, num_updates=1):
    """Prior function for variational layer."""
    n = kernel_size + bias_size
    c = np.log(np.expm1(1e-5))

    def regularizer(t):
        out = tfd.LogNormal(0., 1.).log_prob(1e-5 + tf.nn.softplus(c + t[Ellipsis, -1]))
        return -tf.reduce_sum(out) / num_updates

    # Include the prior on the scale parameter as a regularizer in the loss.
    variable_layer = tfpl.VariableLayer(n, dtype=dtype, regularizer=regularizer)

    def distribution_fn(t):
        scale = 1e-5 + tf.nn.softplus(c + t[Ellipsis, -1])
        return tfd.Independent(tfd.Normal(loc=t[Ellipsis, :n], scale=scale),
                               reinterpreted_batch_ndims=1)

    distribution_layer = tfpl.DistributionLambda(distribution_fn)
    return tf.keras.Sequential([variable_layer, distribution_layer])

def ResNet50_llvi(input_shape, classes, num_updates):
        # Determine proper input shape
    if backend.image_data_format() == 'channels_first':
#        input_shape = (3, 224, 224)
        bn_axis = 1
    else:
#        input_shape = (224, 224, 3)
        bn_axis = 3
    img_input = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)

    # Stage 1
    x = layers.Conv2D(64, (7, 7), use_bias=False,
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis,
                                  momentum=BATCH_NORM_DECAY,
                                  epsilon=BATCH_NORM_EPSILON,
                                  name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # Stage 2
    x = conv_block_base(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block_base(x, 3, [64, 64, 256], stage=2, block='c')
    
    # Stage 3
    x = conv_block_base(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block_base(x, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    x = conv_block_base(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block_base(x, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    x = conv_block_base(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block_base(x, 3, [512, 512, 2048], stage=5, block='c')
    
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)

    # Outer layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
#    x = tfpl.DenseVariational(
#        units=classes,
#        make_posterior_fn=posterior_mean_field,
#        make_prior_fn=functools.partial(
#            prior_trainable, num_updates=num_updates),
#        use_bias=True,
#        kl_weight=1./num_updates,
#        kl_use_exact=True
#        )(x)
    x = tfpl.DenseFlipout(classes)(x)
    x = layers.Activation('softmax')(x)
    
    model = models.Model(img_input, x, name='resnet50')
    
    return model

    

