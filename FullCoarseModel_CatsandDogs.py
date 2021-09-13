# -*- coding: utf-8 -*-
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
import cv2
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns


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


# learning rate decay
STEP = 0


def lr_scheduler(epoch, lr):
    global STEP
    decay_rate = 0.75
    decay_step = [30, 50, 70, 90]
    if STEP < 4 and epoch == decay_step[STEP]:
        STEP += 1
        return lr * decay_rate
    return lr


# custom layers
def custom_argmax_bounding(predictions):
    pred_box = predictions

    t = 53

    bigbox = pred_box[0]
    for i in range(1, (len(pred_box) - 1)):
        b1_x1 = bigbox[:, 0]
        b1_x2 = bigbox[:, 2]
        b1_y1 = bigbox[:, 1]
        b1_y2 = bigbox[:, 3]

        nextbox = pred_box[i]
        b2_x1 = nextbox[:, 0]
        b2_x2 = nextbox[:, 2]
        b2_y1 = nextbox[:, 1]
        b2_y2 = nextbox[:, 3]

        area1 = abs(b1_x1 - b1_x2) * abs(b1_y1 - b1_y2)
        area2 = abs(b2_x1 - b2_x2) * abs(b2_y1 - b2_y2)

        x_dist = tf.math.minimum(b1_x2, b2_x2) - tf.math.maximum(b1_x1, b2_x1)
        y_dist = tf.math.minimum(b1_y2, b2_y2) - tf.math.maximum(b1_y1, b2_y1)

        areaI = 0
        cond_x = tf.greater(x_dist, 0)
        cond_y = tf.greater(y_dist, 0)
        cond = tf.logical_and(cond_x, cond_y)

        area = (x_dist * y_dist)
        area_fudged = tf.fill(tf.shape(area), float(t))
        areaI = tf.where(cond, area, area_fudged)

        match_score = areaI + (x_dist ** 2 + y_dist ** 2) ** (0.5)

        is_valid = tf.less(match_score, t)
        is_valid = tf.stack([is_valid, is_valid, is_valid, is_valid], axis=1)
        el0 = (b1_x1 + b2_x1) / 2
        el1 = (b1_y1 + b2_y1) / 2
        el2 = (b1_x2 + b2_x2) / 2
        el3 = (b1_y2 + b2_y2) / 2

        bigbox_modded = tf.stack([el0, el1, el2, el3], axis=1)
        bigbox = tf.where(is_valid, bigbox_modded, bigbox)
    return bigbox


# models
def create_feature_model(input_shape):
    layer1_conv = tf.keras.layers.Conv2D(filters=96, kernel_size=11, kernel_regularizer=l2(1e-5), strides=4,
                                         activation='relu')
    layer1_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    layer2_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=5, kernel_regularizer=l2(1e-5), strides=1,
                                         activation='relu')
    layer2_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)
    layer3_conv = tf.keras.layers.Conv2D(filters=512, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         padding='same', activation='relu')
    layer4_conv = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         padding='same', activation='relu')
    layer5_conv = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, kernel_regularizer=l2(1e-5), strides=1,
                                         padding='same', activation='relu')
    layer5_max = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)

    x = layer1_conv(input_shape)
    x = layer1_max(x)
    x = layer2_conv(x)
    x = layer2_max(x)
    x = layer3_conv(x)
    x = layer4_conv(x)
    x = layer5_conv(x)
    x = layer5_max(x)

    output = x
    model = Model(inputs=input_shape, outputs=output, name="feature_model")
    return model


def create_classifier_model(input_shape, num_classes):
    layer6_flt = tf.keras.layers.Flatten()
    layer6_dns = tf.keras.layers.Dense(3072, kernel_regularizer=l2(1e-5), activation='relu')  # 3072
    layer6_drp = tf.keras.layers.Dropout(0.5)
    layer7_dns = tf.keras.layers.Dense(4096, kernel_regularizer=l2(1e-5), activation='relu')  # 4096
    layer7_drp = tf.keras.layers.Dropout(0.5)
    layer8_dns = tf.keras.layers.Dense(num_classes, activation='softmax')

    x = layer6_flt(input_shape)
    x = layer6_dns(x)
    x = layer6_drp(x)
    x = layer7_dns(x)
    x = layer7_drp(x)
    x = layer8_dns(x)

    output = x
    model = Model(inputs=input_shape, outputs=output, name="classifier_model")
    return model


def create_regressive_model(input_shape):
    layer6_flt = tf.keras.layers.Flatten()
    layer6_dns = tf.keras.layers.Dense(4096, kernel_regularizer=l2(1e-5), activation='relu')  # 4096
    layer6_drp = tf.keras.layers.Dropout(0.5)
    layer7_dns = tf.keras.layers.Dense(1024, kernel_regularizer=l2(1e-5), activation='relu')  # 1024
    layer7_drp = tf.keras.layers.Dropout(0.5)
    layer8_dns = tf.keras.layers.Dense(4, activation='relu')

    x = layer6_flt(input_shape)
    x = layer6_dns(x)
    x = layer6_drp(x)
    x = layer7_dns(x)
    x = layer7_drp(x)
    x = layer8_dns(x)

    output = x
    model = Model(inputs=input_shape, outputs=output, name="regressive_model")
    return model


def create_classification_model(input_shape, feature_model, classifer_model):
    layer_feature_model = feature_model
    layer_classifer_model = classifer_model
    layer_max = tf.keras.layers.Maximum()

    feature_predictions = []
    for input in input_shape:
        x = layer_feature_model(input)
        x = layer_classifer_model(x)
        feature_predictions.append(x)

    pred = layer_max(feature_predictions)
    output = pred
    model = Model(inputs=input_shape, outputs=output, name="classification_model")
    return model


def create_regression_model(input_shape, feature_model, regressive_model):
    layer_feature_model = feature_model
    layer_regressive_model = regressive_model
    layer_argmax = tf.keras.layers.Lambda(custom_argmax_bounding)

    feature_box_predictions = []
    for input in input_shape:
        x = layer_feature_model(input)
        feat_box_pred = layer_regressive_model(x)
        feature_box_predictions.append(feat_box_pred)

    output_box = layer_argmax(feature_box_predictions)
    model = Model(inputs=input_shape, outputs=output_box, name="regression_model")
    return model


# drawing
def drawBox(boxes, image):
    for i in range(0, len(boxes)):
        cv2.rectangle(image, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), (0, 0, 120), thickness=3)
    im_pil = Image.fromarray(image)
    im_pil.show()


def run():
    # training
    input_train = np.load("/content/drive/MyDrive/School Work/CS/DL/Final Project/PetsData/trainval_X_formatted.npy",
                          allow_pickle=True)
    regression_target_train = np.load(
        "/content/drive/MyDrive/School Work/CS/DL/Final Project/PetsData/trainval_B_formatted.npy", allow_pickle=True)
    classification_hard_target_train = np.load(
        "/content/drive/MyDrive/School Work/CS/DL/Final Project/PetsData/trainval_Y_37_formatted.npy",
        allow_pickle=True)
    classification_hard_target_train = tf.keras.utils.to_categorical(classification_hard_target_train, num_classes=37)
    input_train = sliding_window(np.array(resizeimages(input_train), dtype=object))

    # classification
    input_shape = []
    for i in range(10):
        input_shape.append(Input(shape=input_train[i][0].shape))

    feature_model = create_feature_model(Input(shape=(231, 231, 3)))
    classifier_model = create_classifier_model(Input(shape=(6, 6, 1024)), 37)
    classification_model = create_classification_model(input_shape, feature_model, classifier_model)
    classification_model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=1e-3, momentum=0.6),
                                 metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    callbacks = [LearningRateScheduler(lr_scheduler, verbose=1)]
    history = History()

    classification_model.fit(input_train, classification_hard_target_train, batch_size=64, epochs=100, verbose=1,
                             validation_split=0.2, callbacks=[checkpointer, callbacks, history])

    # classification_model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/Classification_Pets37')
    # feature_model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/Feature_Pets37')
    # classifier_model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/Classifier_Pets37')

    # localization
    for layer in feature_model.layers:
        layer.trainable = False;

    for layer in classifier_model.layers:
        layer.trainable = False;

    regressive_model = create_regressive_model(Input(shape=(6, 6, 1024)))
    regression_model = create_regression_model(input_shape, feature_model, regressive_model)
    regression_model.compile(loss='mean_squared_logarithmic_error', optimizer=SGD(learning_rate=1e-3, momentum=0.6),
                             metrics=['accuracy'])
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)
    history = History()

    regression_model.fit(input_train, regression_target_train, batch_size=64, epochs=200, verbose=1,
                         validation_split=0.2,
                         callbacks=[checkpointer, history])

    # regression_model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/Regression_m_Pets37')
    # regressive_model.save('/content/drive/MyDrive/School Work/CS/DL/Final Project/Models/Regressive_m_Pets37')

    # testing
    whole_images = np.load("/content/drive/MyDrive/School Work/CS/DL/Final Project/PetsData/test_X_formatted.npy",
                         allow_pickle=True)
    target_test = np.load("/content/drive/MyDrive/School Work/CS/DL/Final Project/PetsData/test_Y_37_formatted.npy",
                          allow_pickle=True)
    input_test = sliding_window(whole_images[0:2100])
    target_test = tf.keras.utils.to_categorical(target_test, num_classes=37)[0:2100]

    np.set_printoptions(threshold=np.inf)
    classification_model.evaluate(input_test, target_test)

    # confusion matrix
    sns.set_theme()
    y_pred = classification_model.predict(input_test)
    out = tf.convert_to_tensor(y_pred)
    topk = tf.nn.top_k(out, k=1, sorted=False)
    res = tf.reduce_sum(tf.one_hot(topk.indices, out.get_shape().as_list()[-1]), axis=1)
    plt.rcParams['figure.figsize'] = (20, 16)
    confusion_data = confusion_matrix(tf.argmax(target_test, axis=1), tf.argmax(res, axis=1))
    heat_map = sns.heatmap(confusion_data, cmap='Blues', annot=True, linewidths=0, fmt='g')
    plt.xlabel("Predicted Values")
    plt.ylabel("True Values")
    plt.title("Confusion Matrix")
    plt.show()

    # bounding boxes
    y_pred = regression_model.predict(input_test)
    for i in range(100):
        drawBox([y_pred[i]], whole_images[i])
