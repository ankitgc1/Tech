
#===============================Art by Ankit===============================#

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Input, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
import cv2
import os

a = cv2.imread("data/train/box/1440.jpg")
print(a.shape)  #image shape = (480, 854, 3)

# img_height = int(a.shape[0])
# img_width = int(a.shape[1])
image_size = (255, 255)
image_reshape = (1, 255, 255, 3)

label = []
input_images = []
output_images = []
for filename in os.listdir("data/train/box"):
    # load image
    img_data = cv2.imread("data/train/box/" + filename)
    img_data = tf.image.resize(img_data, image_size)
    img_data = tf.reshape(img_data, image_reshape)
    # store loaded image
    input_images.append(img_data)
    output_images.append(img_data)
    output_images.append(img_data)
    label.append(0)
    if len(input_images)>800:
        break

print(len(input_images), len(output_images), len(label))

for filename in os.listdir("data/train/no_box"):
    # load image
    img_data = cv2.imread("data/train/no_box/"+ filename)
    img_data = tf.image.resize(img_data, image_size)
    img_data = tf.reshape(img_data, image_reshape)
    # store loaded image
    input_images.append(img_data)
    label.append(1)
    if len(input_images)>800:
        break

print(len(input_images), len(output_images), len(label))

# img1 = cv2.imread("data/box/1440.jpg")
# img2 = cv2.imread("data/box/1441.jpg")
# print(img1.shape)
# print(img2.shape)
# image_size = (255, 255)
# img1 = tf.image.resize(img1, image_size)
# img2 = tf.image.resize(img2, image_size)
# nb_classes = 10
"""
    Model architecture
"""
input_shape = (255, 255, 3)
# Define the tensors for the two input images
input_data = Input(shape=input_shape)
output_data = Input(shape=input_shape)

# Convolutional Neural Network
# model = Sequential()
num_classes = 2
label = tf.keras.utils.to_categorical(label, num_classes)
base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
# remove the output layer
# model.layers.pop()
# model = Model(inputs=base_model.inputs, outputs=base_model.layers[-1].output)

model = Flatten()(base_model.output)
model = Dense(4096, activation='relu')(model)
encoded_l = model(input_data)
encoded_r = model(output_data)
model = Model(inputs=[input_data,output_data], outputs=model)
net.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
# get extracted features
# features = model.predict(input_images[0])
print(features.shape)
