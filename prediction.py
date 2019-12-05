
#===============================Art by Ankit===============================#

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import cv2


model = load_model('mk1.h5')

model.compile(loss='binary_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])

img1 = cv2.imread("data/box/1440.jpg")
img2 = cv2.imread("data/box/1441.jpg")
print(img1.shape)
print(img2.shape)
image_size = (255, 255)
img1 = tf.image.resize(img1, image_size)
img2 = tf.image.resize(img2, image_size)
image_reshape = (1, 255, 255, 3)
img1 = tf.reshape.reshape(img1, image_reshape)
img2 = tf.reshape.reshape(img2, image_reshape)
classes = model.predict([img1, img2])

print(classes)
