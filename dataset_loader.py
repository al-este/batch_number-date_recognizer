# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 12:48:51 2021

@author: alest
"""
import numpy as np
import cv2
from skimage import io
import os

from keras.utils import to_categorical

import matplotlib.pyplot as plt

x_train_pre = []
y_train_pre = []

folders = os.listdir("English/fnt")

for d in folders:
    images = io.ImageCollection("English/fnt/" + d + "/*.png")
    
    images = list(images)
    
    x_train_pre.extend(images)

    y = np.zeros(len(images))
    y.fill(int(d[-2:])-1)
    
    y_train_pre.extend(list(y))
    
    print(int(d[-2:]))
    print(len(list(y)))
    
print('-------')
print(len(x_train_pre))
print(len(y_train_pre))


l = int(len(x_train_pre)/10)

x_train = []
y_train = []
for image, y in zip(x_train_pre, y_train_pre):
    image = np.array(image)
    i = cv2.resize(image, dsize=(71,71))
    
    if y == 36:
        x_train.append(-(i-255))
        y_train.append(y)
        x_train.append(i)
        y_train.append(y)
        
        image = image[:, 20:100]
        
    i = cv2.resize(image, dsize=(71,71))
    
    x_train.append(-(i-255))
    y_train.append(y)
    x_train.append(i)
    y_train.append(y)
    
print('\n',y_train[-1])
plt.imshow(x_train[-1])
plt.show()
    
zeros = np.zeros((l, 71, 71))
x_train.extend(list(zeros))

y = np.zeros(l)
y.fill(37)
y_train.extend(list(y))
    
print(len(x_train))
print(len(y_train))

x_train = np.array(x_train, dtype='uint8')
y_train = np.array(y_train, dtype='uint8')

y_train = to_categorical(y_train, 38)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)
input()