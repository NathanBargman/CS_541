import pandas as pd
import numpy as np
from PIL import Image
import os
import math
import untangle
import cv2

def findimgscale(img):
    x = img.size[0]
    y = img.size[1]

    if x > y:
        scalefactor = 256/y
    else:
        scalefactor = 256/x

    return [math.floor(x*scalefactor), math.floor(y*scalefactor)]

def resize(img, box):

    orig_shape = img.size

    scale = findimgscale(img)
    img_resized = img.resize((scale[0], scale[1]), Image.BILINEAR)

    box_scale = scale[0] / orig_shape[0]
    box_resized = [math.floor(box[0] * box_scale),
                   math.floor(box[1] * box_scale),
                   math.floor(box[2] * box_scale),
                   math.floor(box[3] * box_scale)]

    return img_resized, box_resized


def drawBox(boxes, image, name):
    for i in range (0, len(boxes)):
        cv2.rectangle(image,(boxes[i][0],boxes[i][1]),(boxes[i][2],boxes[i][3]),(0,0,120), thickness=3)
    cv2.imwrite(name,image)

def open_and_format_data(file):
    data = pd.read_csv("dataset/annotations/{0}.csv".format(file), delimiter=" ")

    X = []
    Y_2 = []
    Y_37 = []
    B = []

    skipped = 0

    for i, row in data.iterrows():
        img = Image.open("dataset/images/{0}.jpg".format(row["name"]))

        try:
            xml = untangle.parse("dataset/annotations/xmls/{0}.xml".format(row["name"]))
            bndbox = xml.annotation.object.bndbox
            box = [int(bndbox.xmin.cdata), int(bndbox.ymin.cdata), int(bndbox.xmax.cdata), int(bndbox.ymax.cdata)]

            img_resized, box_resized = resize(img, box)
        except:
            skipped += 1
            continue

        drawBox([box], np.array(img), "Orig.png")
        drawBox([box_resized], np.array(img_resized), "Resized.png")

        X.append(np.array(img_resized))
        Y_2.append(row["species"] - 1)
        Y_37.append(row["class"] - 1)
        B.append(box_resized)

    print(skipped)

    X = np.array(X)
    Y_2 = np.array(Y_2)
    Y_37 = np.array(Y_37)
    B = np.array(B)

    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y_2 = Y_2[shuffler]
    Y_37 = Y_37[shuffler]
    B = B[shuffler]

    np.save("{0}_X_formatted.npy".format(file), X)
    np.save("{0}_Y_2_formatted.npy".format(file), Y_2)
    np.save("{0}_Y_37_formatted.npy".format(file), Y_37)
    np.save("{0}_B_formatted.npy".format(file), B)

open_and_format_data("trainval")

# open_and_format_data("test")