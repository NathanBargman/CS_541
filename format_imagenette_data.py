import numpy as np
import os
import cv2 as cv
from PIL import Image


def compile_data(path):
    images = []
    for _, dirs, _ in os.walk("{0}".format(path)):
        for dir in dirs:
            folder = []
            for _, _, files in os.walk("{0}/{1}".format(path, dir)):
                print("{0}/{1}".format(path, dir))
                i = 0
                for file in files:
                    I = Image.open("{0}/{1}/{2}".format(path, dir, file))
                    numpy_img = np.array(I)
                    image = cv.cvtColor(numpy_img, cv.COLOR_RGB2BGR)
                    folder.append(image)
                    I.close()

                    i += 1
                    if i > 500:
                        break
            images.append(folder)
    return images

def generate_labels(images):
    labels = np.zeros(len(images), dtype=object)
    for i, image_set in enumerate(images):
        new_set = np.full(len(image_set), i)
        labels[i] = new_set
    return labels

def flatten_data(data):
    flattened = []
    for element in data:
        for sub_element in element:
            flattened.append(sub_element)
    return np.array(flattened)

def convert_hot_vector(y):
    converted = np.zeros((len(y), np.max(y) - np.min(y) + 1))
    for i in range(len(y)):
        converted[i, y[i]] = 1
    return converted

def generate_and_randomize_data():
    images = compile_data("dataset/imagenette2/train")

    # Generate Labels
    labels = generate_labels(images)

    # Flatten Data
    X = flatten_data(images)
    Y = convert_hot_vector(flatten_data(labels))

    # Randomize data
    shuffler = np.random.permutation(len(X))
    X = X[shuffler]
    Y = Y[shuffler]

    # Split Test Data
    split = int(len(X)/5)
    testX = X[0:split]
    testY = Y[0:split]

    trainX = X[split:-1]
    trainY = Y[split:-1]

    return testX, testY, trainX, trainY



testX, testY, trainX, trainY = generate_and_randomize_data()
np.save("dataset/imagenette.npy", [testX, testY, trainX, trainY])

testX, testY, trainX, trainY = np.load("dataset/imagenette.npy", allow_pickle=True)

print(len(testX))
print(len(testY))
print(len(trainX))
print(len(trainY))