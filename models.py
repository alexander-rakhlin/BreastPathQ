# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import

from keras.layers import Input
from keras import layers
from keras.layers import Dense, Flatten
from keras.layers import Activation

from keras.layers import Conv2D, Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.layers import ELU, Dropout, SpatialDropout2D
from keras.models import Model
from keras import backend as K
from keras.optimizers import Adam

from keras.applications.xception import Xception

import numpy as np

kernel_initializer = "he_normal"


def binary_crossentropy_weighted(y_true, y_pred, class_weights=(1, 1, 1, 1)):
    assert K.ndim(y_true) == K.ndim(y_pred) == 4
    if K.image_data_format() == "channels_first":
        plane_ax = [2, 3]
    else:
        plane_ax = [1, 2]
    # K.binary_crossentropy keeps dimensions
    loss = K.mean(K.binary_crossentropy(y_true, y_pred), axis=plane_ax)   # b x c
    loss = K.sum(loss * class_weights, axis=-1) / sum(class_weights)
    return loss


def jaccard(y_true, y_pred, class_weights=(1, 1, 1, 1)):
    assert K.ndim(y_true) == K.ndim(y_pred) == 4
    if K.image_data_format() == "channels_first":
        plane_ax = [2, 3]
    else:
        plane_ax = [1, 2]

    eps = 1e-15
    intersection = K.sum(y_true * y_pred, axis=plane_ax)
    sum_ = K.sum(y_true + y_pred, axis=plane_ax)
    jcrd = (intersection + eps) / (sum_ - intersection + eps)
    jcrd = K.sum(jcrd * class_weights, axis=-1) / sum(class_weights)
    return jcrd


def jaccard_discrete(y_true, y_pred, class_weights=(1, 1, 1, 1), threshold=0.5):
    assert K.ndim(y_true) == K.ndim(y_pred) == 4
    if K.image_data_format() == "channels_first":
        plane_ax = [2, 3]
    else:
        plane_ax = [1, 2]

    eps = 1e-15
    y_pred = K.cast(K.greater_equal(y_pred, threshold), K.floatx())
    intersection = K.sum(y_true * y_pred, axis=plane_ax)
    sum_ = K.sum(y_true + y_pred, axis=plane_ax)
    jcrd = (intersection + eps) / (sum_ - intersection + eps)
    jcrd = K.sum(jcrd * class_weights, axis=-1) / sum(class_weights)
    return jcrd


def crossentropy_jaccard(y_true, y_pred, class_weights=(1, 1, 1, 1), jaccard_weight=0.5):
    loss_bce = binary_crossentropy_weighted(y_true, y_pred, class_weights=class_weights)
    loss_jac = 1 - jaccard(y_true, y_pred, class_weights=class_weights)
    loss = loss_bce * (1 - jaccard_weight) + loss_jac * jaccard_weight
    return loss


# --- ResNet
#

def decoder_block(input_tensors, filters, stage):
    decoder_name_base = "decoder_" + str(stage)
    filters1, filters2 = filters
    if K.image_data_format() == "channels_last":
        cat_axis = 3
    else:
        cat_axis = 1
    if isinstance(input_tensors, (tuple, list)):
        # x = layers.concatenate(input_tensors, axis=cat_axis)
        x = layers.add(input_tensors)
    else:
        x = input_tensors
    x = UpSampling2D()(x)
    x = Conv2D(filters1, (3, 3), padding="same", kernel_initializer=kernel_initializer,
               name=decoder_name_base + "_conv1")(x)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=cat_axis)(x)
    x = Conv2D(filters2, (3, 3), padding="same", kernel_initializer=kernel_initializer,
               name=decoder_name_base + "_conv2")(x)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=cat_axis)(x)
    return x


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: "a","b"..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"
    output_name = "output" + str(stage) + block

    x = Conv2D(filters1, (3, 3), padding="same", kernel_initializer=kernel_initializer, name=conv_name_base + "2a")(
        input_tensor)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)

    x = Conv2D(filters2, kernel_size, padding="same", kernel_initializer=kernel_initializer,
               name=conv_name_base + "2b")(x)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)

    shortcut = Activation("elu")(input_tensor)
    x = layers.add([x, shortcut])
    x = Activation("elu", name=output_name)(x)

    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: "a","b"..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2 = filters
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    x = Conv2D(filters1, (3, 3), padding="same", strides=strides, kernel_initializer=kernel_initializer,
               name=conv_name_base + "2a")(input_tensor)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2a")(x)

    x = Conv2D(filters2, kernel_size, padding="same", kernel_initializer=kernel_initializer,
               name=conv_name_base + "2b")(x)
    x = Activation("elu")(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + "2b")(x)

    shortcut = Conv2D(filters2, (3, 3), padding="same", strides=strides, kernel_initializer=kernel_initializer,
                      name=conv_name_base + "1")(input_tensor)
    shortcut = Activation("elu")(shortcut)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + "1")(shortcut)

    x = layers.add([x, shortcut])
    x = Activation("elu")(x)

    return x


def ResNet34(include_top=True, input_shape=None, weights_path=None, classes=1000):
    img_input = Input(shape=input_shape)
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer, name="conv1")(
        img_input)
    x = BatchNormalization(axis=bn_axis, name="bn_conv1")(x)
    x = Activation("elu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = conv_block(x, 3, [64, 64], stage=2, block="a", strides=(1, 1))
    x = identity_block(x, 3, [64, 64], stage=2, block="b")
    x = identity_block(x, 3, [64, 64], stage=2, block="c")

    x = conv_block(x, 3, [128, 128], stage=3, block="a")
    x = identity_block(x, 3, [128, 128], stage=3, block="b")
    x = identity_block(x, 3, [128, 128], stage=3, block="c")
    x = identity_block(x, 3, [128, 128], stage=3, block="d")

    x = conv_block(x, 3, [256, 256], stage=4, block="a")
    x = identity_block(x, 3, [256, 256], stage=4, block="b")
    x = identity_block(x, 3, [256, 256], stage=4, block="c")
    x = identity_block(x, 3, [256, 256], stage=4, block="d")
    x = identity_block(x, 3, [256, 256], stage=4, block="e")
    x = identity_block(x, 3, [256, 256], stage=4, block="f")

    x = conv_block(x, 3, [512, 512], stage=5, block="a")
    x = identity_block(x, 3, [512, 512], stage=5, block="b")
    x = identity_block(x, 3, [512, 512], stage=5, block="c")

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    if include_top:
        x = Dense(classes, activation="softmax", kernel_initializer=kernel_initializer, name="fc" + str(classes))(x)

    # Create model.
    model = Model(img_input, x, name="resnet34")

    # load weights
    if weights_path is not None:
        print("Load weights from", weights_path)
        model.load_weights(weights_path)

    optimizer = Adam()
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def uResNet34(input_size=None, encoder_weights=None, weights=None, n_classes=4, jaccard_weight=0.5,
              threshold=0.5, class_weights=(1, 0, 0, 0), training=None):
    assert any([encoder_weights is None, weights is None])  # do not load both weights

    if K.image_data_format() == "channels_last":
        input_shape = input_size + (3,)
    else:
        input_shape = (3,) + input_size

    img_input = Input(shape=input_shape)

    model = ResNet34(include_top=True, input_shape=input_shape, weights_path=encoder_weights, classes=n_classes)
    if encoder_weights is not None:
        print("Freeze encoder")
        for layer in model.layers:
            layer.trainable = False

    model(img_input)

    stage_2 = model.get_layer("output2c").output
    stage_3 = model.get_layer("output3d").output
    stage_4 = model.get_layer("output4f").output
    stage_5 = model.get_layer("output5c").output

    x = decoder_block(stage_5, (256, 256), stage=5)
    x = decoder_block([x, stage_4], (128, 128), stage=4)
    x = decoder_block([x, stage_3], (64, 64), stage=3)
    x = decoder_block([x, stage_2], (64, 64), stage=2)

    x = decoder_block(x, (64, 32), stage=1)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer=kernel_initializer)(x)
    x = Activation("elu")(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer=kernel_initializer)(x)
    x = Activation("sigmoid")(x)

    # Create model.
    model = Model(model.get_input_at(0), x, name="uResNet34")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights, by_name=True)

    optimizer = Adam()
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    def jaccard_loss(y_true, y_pred):
        return crossentropy_jaccard(y_true, y_pred, class_weights=class_weights, jaccard_weight=jaccard_weight)

    def jaccard(y_true, y_pred):
        return jaccard_discrete(y_true, y_pred, class_weights=class_weights, threshold=threshold)

    model.compile(loss=jaccard_loss, optimizer=optimizer, metrics=[jaccard])

    return model


def fpn_block(upsampling_factor, stage, filters=64):
    fpn_name_base = "fpn_" + str(stage)
    if K.image_data_format() == "channels_last":
        cat_axis = 3
    else:
        cat_axis = 1

    def f(x):
        x = Conv2D(filters, (3, 3), padding="same", kernel_initializer=kernel_initializer,
                   name=fpn_name_base + "_conv1")(x)
        if upsampling_factor > 1:
            x = UpSampling2D(upsampling_factor)(x)
        # x = Activation("elu")(x)
        # x = BatchNormalization(axis=cat_axis)(x)
        # x = Conv2D(filters2, (3, 3), padding="same", kernel_initializer=kernel_initializer,
        #            name=decoder_name_base + "_conv2")(x)
        # x = Activation("elu")(x)
        # x = BatchNormalization(axis=cat_axis)(x)
        return x
    return f


def uResNet34FPN(input_shape=None, encoder_weights=None, weights=None, n_classes=4, jaccard_weight=0.5,
                 threshold=0.5, class_weights=(1, 0, 0, 0), training=None):
    assert any([encoder_weights is None, weights is None])  # do not load both weights
    if K.image_data_format() == "channels_last":
        cat_axis = 3
    else:
        cat_axis = 1

    img_input = Input(shape=input_shape)

    model = ResNet34(include_top=True, input_shape=input_shape, weights_path=encoder_weights, classes=n_classes)
    if encoder_weights is not None:
        print("Freeze encoder")
        for layer in model.layers:
            layer.trainable = False

    model(img_input)

    stage_2 = model.get_layer("output2c").output
    stage_3 = model.get_layer("output3d").output
    stage_4 = model.get_layer("output4f").output
    stage_5 = model.get_layer("output5c").output

    x5 = decoder_block(stage_5, (256, 256), stage=5)
    x4 = decoder_block([x5, stage_4], (128, 128), stage=4)
    x3 = decoder_block([x4, stage_3], (64, 64), stage=3)
    x2 = decoder_block([x3, stage_2], (64, 64), stage=2)

    x = layers.concatenate([
        fpn_block(8, 5)(x5),
        fpn_block(4, 4)(x4),
        fpn_block(2, 3)(x3),
        fpn_block(1, 2)(x2),
    ], axis=cat_axis)

    # x = SpatialDropout2D(rate=0.7)(x)
    x = SpatialDropout2D(rate=0.3)(x, training=training)

    x = decoder_block(x, (64, 32), stage=1)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer=kernel_initializer)(x)
    x = Activation("elu")(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer=kernel_initializer)(x)
    x = Activation("sigmoid")(x)

    # Create model.
    model = Model(model.get_input_at(0), x, name="uResNet34FPN")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights)

    optimizer = Adam(decay=1e-4)
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    def jaccard_loss(y_true, y_pred):
        return crossentropy_jaccard(y_true, y_pred, class_weights=class_weights, jaccard_weight=jaccard_weight)

    def jaccard(y_true, y_pred):
        return jaccard_discrete(y_true, y_pred, class_weights=class_weights, threshold=threshold)

    model.compile(loss=jaccard_loss, optimizer=optimizer, metrics=[jaccard])

    return model


# --- VGG
#

def vgg_block(num_filters, block_num, sz=3):
    def f(input_):
        if K.image_data_format() == "channels_last":
            bn_axis = 3
        else:
            bn_axis = 1
        b = "block" + str(block_num)
        x = Convolution2D(num_filters, (sz, sz), kernel_initializer=kernel_initializer, padding="same",
                          name=b + "_conv1")(input_)
        x = ELU(name=b + "_elu1")(x)
        x = BatchNormalization(axis=bn_axis, name=b + "_bn1")(x)
        x = Convolution2D(num_filters, (1, 1), kernel_initializer=kernel_initializer, name=b + "_conv2")(x)
        x = ELU(name=b + "_elu2")(x)
        x = BatchNormalization(axis=bn_axis, name=b + "_bn2")(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same", name=b + "_pool")(x)
        return x

    return f


def m46(include_top=True, input_shape=None, lr=1e-3, weights_path=None, classes=1000):
    img_input = Input(shape=input_shape)

    # Convolution blocks
    x = vgg_block(16, 1)(img_input)  # Block 1
    x = vgg_block(32, 2)(x)  # Block 2
    x = vgg_block(64, 3)(x)  # Block 3
    x = vgg_block(128, 4)(x)  # Block 4
    x = vgg_block(256, 5)(x)  # Block 5

    x = GlobalAveragePooling2D(name="global_avg_pool")(x)

    if include_top:
        # Classification block
        x = Dropout(0.3, name="dropout1")(x)
        x = Dense(1024, kernel_initializer=kernel_initializer, name="fc1")(x)
        x = ELU()(x)
        x = Dropout(0.3, name="dropout2")(x)
        x = ELU()(x)
        x = Dense(classes, activation="softmax", kernel_initializer=kernel_initializer, name="predictions")(x)

    # Create model.
    model = Model(img_input, x, name="m46")

    # load weights
    if weights_path is not None:
        print("Load weights from", weights_path)
        model.load_weights(weights_path)

    optimizer = Adam()
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    return model


def u46(input_shape=None, encoder_weights=None, weights=None, n_classes=4, jaccard_weight=0.5, threshold=0.5,
        class_weights=(1, 0, 0, 0)):
    assert any([encoder_weights is None, weights is None])  # do not load both weights
    if K.image_data_format() == "channels_last":
        channel_axis = 3
    else:
        channel_axis = 1

    img_input = Input(shape=input_shape)

    model = m46(input_shape=input_shape, weights_path=encoder_weights, classes=2)
    if encoder_weights is not None:
        print("Freeze encoder")
        for layer in model.layers:
            layer.trainable = False

    model(img_input)

    stage_2 = model.get_layer("block2_pool").output
    stage_3 = model.get_layer("block3_pool").output
    stage_4 = model.get_layer("block4_pool").output
    stage_5 = model.get_layer("block5_pool").output
    # stage_6 = model.get_layer("block6_pool").output
    # stage_7 = model.get_layer("block7_pool").output

    # x = decoder_block(stage_7, (512, 384), stage=7)
    # x = decoder_block([x, stage_6], (256, 256), stage=6)
    x = decoder_block(stage_5, (128, 128), stage=5)
    x = decoder_block([x, stage_4], (64, 64), stage=4)
    x = decoder_block([x, stage_3], (32, 32), stage=3)
    x = decoder_block([x, stage_2], (32, 32), stage=2)

    x = decoder_block(x, (32, 32), stage=1)
    x = Conv2D(32, (3, 3), padding="same", kernel_initializer=kernel_initializer)(x)
    x = Activation("elu")(x)
    x = Conv2D(n_classes, (1, 1), kernel_initializer=kernel_initializer)(x)
    x = Activation("sigmoid")(x)

    # Create model.
    model = Model(model.get_input_at(0), x, name="u46")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights)

    optimizer = Adam()
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    def jaccard_loss(y_true, y_pred):
        return crossentropy_jaccard(y_true, y_pred, class_weights=class_weights, jaccard_weight=jaccard_weight)

    def jaccard(y_true, y_pred):
        return jaccard_discrete(y_true, y_pred, class_weights=class_weights, threshold=threshold)

    model.compile(loss=jaccard_loss, optimizer=optimizer, metrics=[jaccard])

    return model


# Regressions
#
def uResNet34regr(input_size=None, unet_weights=None, weights=None, n_classes=4, freeze_unet=False):
    assert any([unet_weights is None, weights is None])  # do not load both weights

    # U-Net
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (3,)
    else:
        input_shape = (3,) + input_size
    model_unet = uResNet34(input_size=input_size, encoder_weights=None,
                           weights=unet_weights, n_classes=n_classes)
    # for layer in model_unet.layers:
    #     layer.name += "_"
    if freeze_unet:
        print("Freeze U-Net")
        for layer in model_unet.layers:
            layer.trainable = False
    img_input = Input(shape=input_shape)
    model_unet(img_input)

    # Regression
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (n_classes,)
    else:
        input_shape = (n_classes,) + input_size
    model = ResNet34(include_top=False, input_shape=input_shape, weights_path=None)
    x = model(model_unet.get_output_at(0))
    x = Dense(1, activation="linear", kernel_initializer=kernel_initializer, name="output")(x)

    # Create model.
    model = Model(model_unet.get_input_at(0), x, name="uResNet34regr")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights)

    optimizer = Adam(decay=1e-4)
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return model


def m46regr(input_shape, num_outputs=1, dropout=0.3, lr=1e-3, weights_path=None):
    img_input = Input(shape=input_shape)

    # Convolution blocks
    x = vgg_block(32, 1)(img_input)  # Block 1
    x = vgg_block(64, 2)(x)  # Block 2
    x = vgg_block(128, 3)(x)  # Block 3
    x = vgg_block(128, 4)(x)  # Block 4
    x = vgg_block(256, 5)(x)  # Block 5
    x = vgg_block(384, 6)(x)  # Block 6

    # Regression block
    x = Flatten(name='flatten')(x)
    x = Dropout(dropout, name='dropout1')(x, training=True)
    x = Dense(2048, kernel_initializer=kernel_initializer, name='fc1')(x)
    x = ELU()(x)
    x = Dropout(dropout, name='dropout2')(x, training=True)
    x = Dense(2048, kernel_initializer=kernel_initializer, name='fc2')(x)
    x = ELU()(x)
    x = Dense(num_outputs, activation='linear', name='predictions')(x)

    # Create model
    model = Model(img_input, x)
    if weights_path:
        print('load weights from', weights_path)
        model.load_weights(weights_path)

    # opt = SGD(lr=lr, momentum=0.95, decay=0.0005, nesterov=True)
    opt = Adam()
    model.compile(loss='mae', optimizer=opt)
    return model


def uResNet34m46regr(input_size=None, unet_weights=None, weights=None, n_classes=4, freeze_unet=False):
    assert any([unet_weights is None, weights is None])  # do not load both weights

    # U-Net
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (3,)
    else:
        input_shape = (3,) + input_size
    model_unet = uResNet34(input_size=input_size, encoder_weights=None,
                           weights=unet_weights, n_classes=n_classes)
    if freeze_unet:
        print("Freeze U-Net")
        for layer in model_unet.layers:
            layer.trainable = False
    img_input = Input(shape=input_shape)
    model_unet(img_input)

    # Regression
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (n_classes,)
    else:
        input_shape = (n_classes,) + input_size
    model = m46regr(input_shape=input_shape, weights_path=None)
    x = model(model_unet.get_output_at(0))

    # Create model.
    model = Model(model_unet.get_input_at(0), x, name="uResNet34m46regr")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights)

    optimizer = Adam(decay=1e-4)
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return model


def uResNet34Xceptionregr(input_size=None, unet_weights=None, weights=None, n_classes=4, freeze_unet=False):
    assert any([unet_weights is None, weights is None])  # do not load both weights

    # U-Net
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (3,)
    else:
        input_shape = (3,) + input_size
    model_unet = uResNet34(input_size=input_size, encoder_weights=None,
                           weights=unet_weights, n_classes=n_classes)
    if freeze_unet:
        print("Freeze U-Net")
        for layer in model_unet.layers:
            layer.trainable = False
    img_input = Input(shape=input_shape)
    model_unet(img_input)

    # Regression
    if K.image_data_format() == "channels_last":
        input_shape = input_size + (n_classes,)
    else:
        input_shape = (n_classes,) + input_size
    model = Xception(input_shape=input_shape, include_top=False, weights=None, pooling="avg")
    x = model(model_unet.get_output_at(0))
    x = Dense(1, activation="linear", name="predictions")(x)

    # Create model.
    model = Model(model_unet.get_input_at(0), x, name="uResNet34Xceptionregr")

    if weights is not None:
        print("Load weights from", weights)
        model.load_weights(weights)

    optimizer = Adam(decay=1e-4)
    # optimizer = SGD(momentum=0.95, decay=0.0005, nesterov=True)

    model.compile(loss="mse", optimizer=optimizer, metrics=["mse"])

    return model


if __name__ == "__main__":
    UNET_WEIGHTS = "dumps/uResNet34.sz256x256j0.15z2.46-0.768.hdf5"
    SIZE = (448, 448)
    model = uResNet34m46regr(input_size=SIZE, unet_weights=UNET_WEIGHTS, weights=None)
    pass