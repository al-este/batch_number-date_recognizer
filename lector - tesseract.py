# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 20:35:36 2021

@author: alest
"""
import cv2
import numpy as np

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Lee los caracteres de la imagen y los devuelve separados por lineas
def read_characters(image):
    text = pytesseract.image_to_string(image)
    
    text = text.splitlines()
    
    return text

# Leemos la imagen
img = cv2.imread('img_1.jpg', cv2.IMREAD_GRAYSCALE)
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

_,contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

coordinates = []
text_lines = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    
    if np.mean(dilated[y:y+h, x:x+w]) > 170:
        x *= 10
        y *= 10
        h *= 10
        w *= 10
        
        # Extraemos el texto de las regiones de mas de cierto tamaño
        if w > 20 and h > 20:
            text = read_characters(img[y:y+h, x:x+w])
            
            for line in text:
                line = line.strip()
                if len(line) >=2:
                    text_lines.append(line)
    
print(text_lines)


input('\n Presiona enter para salir')