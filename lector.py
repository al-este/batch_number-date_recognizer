# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:35:36 2021

@author: alest
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot(image):
    plt.imshow(image, cmap='Greys')
    plt.show()
    
def read_characters(image):
    width = np.shape(image)[1]
    height = np.shape(image)[0]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # image = cv2.GaussianBlur(image,(5,5),0)
    
    ret,thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # thresh1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # kernel = np.ones((3,3), np.uint8)
    # dilated = cv2.dilate(thresh1, kernel, iterations = 1)
    dilated = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    plot(image)
    plot(thresh1)
    plot(dilated)
    
    _,contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cordinates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        
        if 5 < w < width/5 and 5 < h < width/5 and 0.3<h/w<3.3:
            cordinates.append((x,y,w,h))
            #bound the images
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),1)
        
    plot(image)


img = cv2.imread('img_2.jpg', cv2.IMREAD_GRAYSCALE)
width = int(img.shape[1] / 10)
height = int(img.shape[0] / 10)
dim = (width, height)
reduced_img = cv2.resize(img, dim)

image = np.array(reduced_img)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(image)

image = cv2.GaussianBlur(image,(5,5),0)

ret,thresh1 = cv2.threshold(image, 127,255,cv2.THRESH_BINARY)

kernel = np.ones((4,4), np.uint8)
# dilated = cv2.dilate(thresh1, kernel, iterations = 2)
# dilated = cv2.erode(dilated, kernel, iterations = 2)
dilated = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations = 2)

_,contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# plot(thresh1)

coordinates = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    # coordinates.append((x,y,w,h))
    #bound the images

    # cv2.rectangle(dilated,(x,y),(x+w,y+h),(127,127,127),1)
    if np.mean(dilated[y:y+h, x:x+w]) > 170:
        coordinates.append((x,y,w,h))
        # cv2.rectangle(reduced_img,(x,y),(x+w,y+h),(255,255,255),1)
        

# plot(dilated)
# plot(reduced_img)
for bound in coordinates:
    x, y, w, h = np.array(bound)*10
    if w > 20 and h > 20:
        # pass
        # plot(img[y:y+h, x:x+w])
        read_characters(img[y:y+h, x:x+w])