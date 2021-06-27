# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 13:27:03 2021

@author: alest
"""
CHARACTERS = ['0','1','2','3','4','5','6','7','8','9',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','/','']
import matplotlib.pyplot as plt
import numpy as np

import cv2
import keras
from keras.preprocessing.image import ImageDataGenerator

x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")
x_train = np.expand_dims(x_train, axis=3)

# print(CHARACTERS[np.argmax(y_train[-1])])
# plt.imshow(x_train[-1], cmap='Greys')
# plt.show()

index=np.arange(np.shape(x_train)[0])
np.random.shuffle(index)
# print(index[0:20])

x_train = x_train[index]
y_train = y_train[index]

# def data_prep(image):
#     # image = np.array(image)
#     noise = np.array(np.random.random((71,71,1))*200, dtype='uint8')
#     mean = np.mean(image[:25, :25])
#     if mean == 0:
#         s = np.random.randint(5, 71)
#         noise = np.array(np.random.random((s,s))*255, dtype='uint8')
#         noise = cv2.resize(noise, dsize=(71,71))
#         noise = np.expand_dims(noise, axis=2)
        
#     image = np.clip(image/2+noise, 0, 255)
    
#     image = cv2.GaussianBlur(image,(7,7),0)
#     image = np.expand_dims(image, axis=2)
    
#     return image

def data_prep(image):
    # image = np.array(image)
    
    noise = np.array(np.random.random((71,71,1))*(190*np.random.random()), dtype='uint8')
    mean = np.mean(image[:25, :25])
    if mean == 0:
        s = np.random.randint(5, 71)
        noise = np.array(np.random.random((s,s))*255, dtype='uint8')
        noise = cv2.resize(noise, dsize=(71,71))
        noise = np.expand_dims(noise, axis=2)
        
    image = np.clip(image/2+noise, 0, 255)
    
    image = cv2.GaussianBlur(image,(7,7),0)
    
    image = np.where(np.array(image, dtype='uint8')<127, 0, 255)
    
    image = np.expand_dims(np.array(image, dtype='float32'), axis=2)
    
    return image

datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            width_shift_range=0.1,
            height_shift_range=0.05,
            shear_range=10,
            rotation_range=10,
            horizontal_flip=False,
            vertical_flip=False,
            zoom_range=(0.8, 1.1),
            fill_mode='reflect',
            preprocessing_function=data_prep)

train_generator = datagen.flow(x_train, y_train, batch_size=5, subset='training')

i = 0
for image, mage in train_generator:
    print(CHARACTERS[np.argmax(mage[0])])
    plt.imshow(image[0], cmap='Greys')
    plt.show()
    i+=1
    if i == 10:
        break