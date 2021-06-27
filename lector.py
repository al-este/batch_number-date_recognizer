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
             'A','B','C','D','E','F','G','H','/','J','K','L','M','N','0','P','Q','R','S','T','U','V','W','X','Y','Z','/','']

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
    
    ret,thresh = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
    
    # Imagen donde se extraen los contornos de las letras
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations = 1)
    
    # plot(image)
    # plot(thresh)
    # plot(dilated)
    
    # Imagen donde se extraen los contornos de los conjuntos de caracteres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    di = cv2.erode(thresh, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    di = cv2.dilate(di, kernel, iterations = 2)


    # Procesado de la imagen para reconocer las letras
    proc_image = image

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9,9))
    proc_image = clahe.apply(proc_image)

    proc_image = cv2.GaussianBlur(proc_image,(13,13),0)
    
    proc_image = cv2.adaptiveThreshold(proc_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,10)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    proc_image =cv2.dilate(proc_image, kernel, iterations = 1)
    proc_image =cv2.erode(proc_image, kernel, iterations = 1)
    
    # plot(proc_image)
    
    detected_text = []
    _,contours_sub, hierarchy_sub = cv2.findContours(di,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt_sub in contours_sub:
        X,Y,W,H = cv2.boundingRect(cnt_sub)
        
        if H < 0.9*height and W < 0.9*width:
            x = int(X - W * 0.15)
            y = int(Y - H * 0.2)
            w = int(W * 1.3)
            h = int(H * 1.4)
            
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
                
                if 10 < W < width/5 and 10 < H < width/5 and 0.3<H/W<3.3 and x > 2 and y > 2:
                    x = int(X - W * 0.1)
                    y = int(Y - H * 0.1)
                    w = int(W * 1.2)
                    h = int(H * 1.2)
                    
                    if not(x >= 0 and y >= 0 and x+w < width and y+h < height):
                        x = X
                        y = Y
                        w = W
                        h = H
                    
                    coordinates.append((x,y,w,h))
                    
                    crop_img = sub_image[y:y+h, x:x+w]
                    crop_img = cv2.resize(crop_img, (71,71))
                    crop_img = np.where(crop_img<127, 0, 255)
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
                detected_text.append(text)
                print(text)
                # plot(image)
                print('------------------------')
                
    return detected_text


# Leemos la imagen
img = cv2.imread('img_2.jpg', cv2.IMREAD_GRAYSCALE)
width = int(img.shape[1] / 10)
height = int(img.shape[0] / 10)

# Creamos una imagen de tamaño reducido para acelerar la busqueda de la zona del lote
dim = (width, height)
reduced_img = cv2.resize(img, dim)

image = np.array(reduced_img)

# Corregimos la iluminacion
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
image = clahe.apply(image)

image = cv2.GaussianBlur(image,(5,5),0)

# Buscamos los contornos de la zona donde se encuentran el lote y la fecha de caducidad
ret,thresh1 = cv2.threshold(image, 127,255,cv2.THRESH_BINARY)

kernel = np.ones((4,4), np.uint8)
dilated = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel, iterations = 2)

_,contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# plot(thresh1)

coordinates = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    epsilon = 0.01*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
    # Se procesan solo los contornos cuadrados
    if np.mean(dilated[y:y+h, x:x+w]) > 170 and len(approx) == 4:
        coordinates.append((x,y,w,h))
        

text_lines = []
for bound in coordinates:
    x, y, w, h = np.array(bound)*10
    
    # Descartamos contornos pequeños
    if w > 20 and h > 20:
        text = read_characters(img[y:y+h, x:x+w])
        
        for line in text:
            if len(line) >=2:
                text_lines.append(line)
    
print(text_lines)