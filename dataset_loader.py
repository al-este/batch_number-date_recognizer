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
y_train = []

folders = os.listdir("English/fnt")

for d in folders:
    images = io.ImageCollection("English/fnt/" + d + "/*.png")
    
    images = list(images)
    
    x_train_pre.extend(images)

    y = np.zeros(len(images)*2)
    y.fill(int(d[-2:])-1)
    
    y_train.extend(list(y))
    
    print(int(d[-2:]))

l = len(x_train_pre)

x_train = []
for image in x_train_pre:
    i = cv2.resize(image, dsize=(71,71))
    x_train.append(-(i-255))
    x_train.append(i)
    
zeros = np.zeros((l, 71, 71))
x_train.extend(list(zeros))

y = np.zeros(l)
y.fill(37)
y_train.extend(list(y))
    

x_train = np.array(x_train, dtype='uint8')
y_train = np.array(y_train, dtype='uint8')

y_train = to_categorical(y_train, 38)

np.save("x_train.npy", x_train)
np.save("y_train.npy", y_train)