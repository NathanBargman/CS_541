import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
import math
import random
import ssl
import scipy.misc as s
from PIL import Image
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, History
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, \
    merge, Reshape, Activation
from keras.optimizers import SGD
from sklearn.metrics import log_loss
from keras.utils import to_categorical
from keras import backend as K

from keras.callbacks import LearningRateScheduler


# preprocessing
def findimgscale(img):
    x = img.shape[1]
    y = img.shape[0]

    scalefactor = 1
    if x > y:
        scalefactor = 256 / y
    else:
        scalefactor = 256 / x
    return [math.floor(x * scalefactor), math.floor(y * scalefactor)]


def resizeimages(imgs):
    imgs_resized = []
    for img in imgs:
        img_resized = []
        scale = findimgscale(img)
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((scale[0], scale[1]), Image.BILINEAR)
        new_img = np.array(pil_img)
        imgs_resized.append(new_img)
    return imgs_resized


def sampleinput_classification(img):
    x = img.shape[0]
    y = img.shape[1]
    offset_x = x - 231
    offset_y = y - 231
    offset_x_2 = offset_x // 2
    offset_y_2 = offset_y // 2
    cuttoff_x = x - offset_x
    cuttoff_y = y - offset_y

    img_0 = img[0:cuttoff_x, 0:cuttoff_y]
    img_1 = img[offset_x:, 0:cuttoff_y]
    img_2 = img[offset_x_2:cuttoff_x + offset_x_2, offset_y_2:cuttoff_y + offset_y_2]
    img_3 = img[0:cuttoff_x, offset_y:]
    img_4 = img[offset_x:, offset_y:]

    return np.array([img_0, np.flip(img_0, axis=1),
                     img_1, np.flip(img_1, axis=1),
                     img_2, np.flip(img_2, axis=1),
                     img_3, np.flip(img_3, axis=1),
                     img_4, np.flip(img_4, axis=1)])


def sliding_window(input):
    output = []

    for l in range(10):
        output.append([])

    for img in input:
        all = sampleinput_classification(img)
        for trans_index in range(10):
            output[trans_index].append(all[trans_index])

    for l in range(10):
        output[l] = np.array(output[l])

    return output


# learning rate
STEP = 0


def lr_scheduler(epoch, lr):
    global STEP
    decay_rate = 0.5
    decay_step = [30, 50, 60, 70, 80]
    if STEP < 5 and epoch == decay_step[STEP]:
        STEP += 1
        return lr * decay_rate
    return lr


# custom layers
def custom_find_maps(input):
    pre_pooled_feature_maps = []

    for x in range(0, 2):
        for y in range(0, 2):
            input_modded = input[:, x:-1, y:-1, :]
            pre_pooled_feature_maps.append(input_modded)

    return pre_pooled_feature_maps


def custom_extract_windows(map):
    windows = []

    for x in range(0, 2):
        for y in range(0, 2):
            window = map[:, x:(5 + x), y:(5 + y), :]
            windows.append(window)

    return windows


# models
def create_windows_model(input_shape):
    layer5_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    layer_findmaps = tf.keras.layers.Lambda(custom_find_maps)
    layer_extract_windows = tf.keras.layers.Lambda(custom_extract_windows)

    pre_pooled_feature_map = layer_findmaps(input_shape)
    pooled_feature_map = []
    for ppfm in pre_pooled_feature_map:
        x = layer5_max(ppfm)
        x = layer_extract_windows(x)
        for map in x:
            pooled_feature_map.append(map)

    output = pooled_feature_map[0:2]
    model = Model(inputs=input_shape, outputs=output, name="window_model")
    return model


def create_classifier_model(input_shape):
    layer6_flt = tf.keras.layers.Flatten()
    layer6_dns = tf.keras.layers.Dense(3072, kernel_regularizer=l2(1e-5), activation='relu')  # 3072
    layer6_drp = tf.keras.layers.Dropout(0.5)
    layer7_dns = tf.keras.layers.Dense(4096, kernel_regularizer=l2(1e-5), activation='relu')  # 4096
    layer7_drp = tf.keras.layers.Dropout(0.5)
    layer8_dns = tf.keras.layers.Dense(10, activation='softmax')

    layer_max = tf.keras.layers.Maximum()

    feature_map_predictions = []
    for input in input_shape:
        x = layer6_flt(input)
        x = layer6_dns(x)
        x = layer6_drp(x)
        x = layer7_dns(x)
        x = layer7_drp(x)
        x = layer8_dns(x)
        feature_map_predictions.append(x)

    output = layer_max(feature_map_predictions)
    model = Model(inputs=input_shape, outputs=output, name="softmax_model")
    return model


def create_model(input_shape):
    layer1_conv = tf.keras.layers.Conv2D(filters=96, kernel_size=11, kernel_regularizer=l2(1e-5), strides=4,
                                         activation='relu')
    layer1_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    layer2_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=5, kernel_regularizer=l2(1e-5), strides=1,
                                         activation='relu')
    layer2_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    layer3_pad = tf.keras.layers.ZeroPadding2D((1, 1))
    layer3_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         padding='same', activation='relu')
    layer4_conv = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         activation='relu')
    layer5_pad = tf.keras.layers.ZeroPadding2D((1, 1))
    layer5_conv = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         padding='same', activation='relu')

    layer_windows_model = create_windows_model(Input(shape=(14, 14, 1024)))

    classifier_layer_shape = []
    for i in range(2):
        classifier_layer_shape.append(Input(shape=(5, 5, 1024)))
    layer_classifier_model = create_classifier_model(classifier_layer_shape)

    layer_max = tf.keras.layers.Maximum()

    feature_map_prediction = []
    for input in input_shape:
        x = layer1_conv(input)
        x = layer1_max(x)
        x = layer2_conv(x)
        x = layer2_max(x)
        x = layer3_pad(x)
        x = layer3_conv(x)
        x = layer4_conv(x)
        x = layer5_pad(x)
        x = layer5_conv(x)

        x = layer_windows_model(x)
        x = layer_classifier_model(x)
        feature_map_prediction.append(x)

    output = layer_max(feature_map_prediction)
    model = Model(inputs=input_shape, outputs=output)
    return model


def run():
    input_test, target_test, input_train, target_train = np.load("imagenette2.npy", allow_pickle=True)
    input_train = sliding_window(np.array(resizeimages(input_train)))
    input_test = sliding_window(np.array(resizeimages(input_test)))

    input_shapes = []
    for i in range(10):
        input_shapes.append(Input(shape=input_train[i][0].shape))

    model = create_model(input_shapes)
    model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=1e-3, momentum=0.6),
                  metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
    history = History()

    model.fit(input_train, target_train, batch_size=32, epochs=200, verbose=1, validation_split=0.2,
              callbacks=[checkpointer, callbacks, history])

    np.set_printoptions(threshold=np.inf)
    model.evaluate(input_test, target_test)

    model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/FirstTwoModel')
