# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:35:36 2021

@author: alest
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import model_from_json

CHARACTERS = ['0','1','2','3','4','5','6','7','8','9',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','/','']

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights('model.h5')

def plot(image):
    plt.imshow(image, cmap='Greys')
    plt.show()
    
def read_characters(image):
    width = np.shape(image)[1]
    height = np.shape(image)[0]
    
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    # image = clahe.apply(image)
    
    # image = cv2.GaussianBlur(image,(5,5),0)
    
    ret,thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # thresh1 = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilated = cv2.dilate(thresh, kernel, iterations = 1)
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    # plot(image)
    # plot(thresh)
    # plot(dilated)
    # -------------------------------------------------------------------------
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    di = cv2.erode(thresh, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    di = cv2.dilate(di, kernel, iterations = 2)
    # di = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    # plot(di)
    
    
    # -------------------------------------------------------------------------
    proc_image = image
    
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15,15))
    # proc_image = clahe.apply(proc_image)
    
    # proc_image = cv2.GaussianBlur(proc_image,(5,5),0)
    
    
    # proc_image = cv2.adaptiveThreshold(proc_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,10)
    
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # proc_image =cv2.dilate(proc_image, kernel, iterations = 1)

    
    # proc_image = cv2.GaussianBlur(proc_image,(5,5),0)
    
    plot(proc_image)
    
    _,contours_sub, hierarchy_sub = cv2.findContours(di,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt_sub in contours_sub:
        X,Y,W,H = cv2.boundingRect(cnt_sub)
        
        if H < 0.9*height and W < 0.9*width:
            x = int(X - W * 0.15)
            y = int(Y - H * 0.25)
            w = int(W * 1.3)
            h = int(H * 1.5)
            
            if not(x >= 0 and y >= 0 and x+w < width and y+h < height):
                x = X
                y = Y
                w = W
                h = H
                
            sub_dilated = dilated[y:y+h, x:x+w]
            sub_image = proc_image[y:y+h, x:x+w]
            # plot(sub_image)
            
            _,contours, hierarchy = cv2.findContours(sub_dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            coordinates = []
            images = []
            for cnt in contours:
                X,Y,W,H = cv2.boundingRect(cnt)
                
                if 5 < W < width/5 and 5 < H < width/5 and 0.3<H/W<3.3:
                    x = int(X - W * 0.15)
                    y = int(Y - H * 0.15)
                    w = int(W * 1.3)
                    h = int(H * 1.3)
                    
                    if not(x >= 0 and y >= 0 and x+w < width and y+h < height):
                        x = X
                        y = Y
                        w = W
                        h = H
                    
                    coordinates.append((x,y,w,h))
                    
                    crop_img = sub_image[y:y+h, x:x+w]
                    crop_img = cv2.resize(crop_img, (71,71))
                    crop_img = np.expand_dims(crop_img,axis=2)
                    
                    # plot(crop_img)
                    
                    images.append(crop_img/255.0)
          
            if len(images) > 0:
                
                res = model.predict(np.array(images), batch_size=32)
                
                (res, coordinates) = zip(*sorted(zip(res, coordinates),
                                                    key=lambda b:b[1][0], reverse=False))
                
                text = []
                for ch, pos in zip(res, coordinates):
                    text.append(CHARACTERS[np.argmax(ch)])
                    # print(CHARACTERS[np.argmax(ch)], pos)
                    
                text = ''.join(text)
                print(text)
                # plot(image)
                print('------------------------')

img = cv2.imread('img_1.jpg', cv2.IMREAD_GRAYSCALE)
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

    if np.mean(dilated[y:y+h, x:x+w]) > 170:
        coordinates.append((x,y,w,h))
        
it = 0

for bound in coordinates:
    x, y, w, h = np.array(bound)*10
    if w > 20 and h > 20 and ( it == 2 or it == 16):
        print(it)
        # plot(img[y:y+h, x:x+w])
        read_characters(img[y:y+h, x:x+w])
    it+=1
    