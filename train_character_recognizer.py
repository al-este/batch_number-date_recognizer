# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:36:16 2021

@author: alest
"""


import matplotlib.pyplot as plt
import numpy as np
import cv2

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import Input, Flatten
from keras.layers import AveragePooling2D as Pooling
# from keras.layers import MaxPooling2D as Pooling

from keras.regularizers import l2
from keras.models import Model

import keras.optimizers as opt

from keras.callbacks import LearningRateScheduler



x_train = np.load("x_train.npy")
y_train = np.load("y_train.npy")

x_train = np.expand_dims(x_train, axis=3)

print(np.shape(x_train))
print(np.shape(y_train))

def net_layer(inputs,
              num_filters=16,
              kernel_size=3,
              strides=1,
              activation='relu',
              batch_normalization=True,
              conv_first=False):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            # pass
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def dense_net(input_shape, depth, num_classes=10, k=1):
    filters = 16

    inputs = Input(shape=input_shape)
    x = net_layer(inputs=inputs, conv_first=True, num_filters=filters)

    for d in range(depth):
        if d != 0:
            x = net_layer(inputs=x,
                             num_filters=filters,
                             activation=None,
                             kernel_size=1
                             )
            x = Pooling(pool_size=4)(x)

        y = net_layer(inputs=x,
                       num_filters=filters)
        layers = [x, y]
        for kn in range(k-1):
            l = keras.layers.add(layers)
            l = net_layer(inputs=l,
                              num_filters=filters)
            layers.append(l)

        x = keras.layers.add(layers)
        x = net_layer(inputs=x,
                          num_filters=filters)
        filters*=2

    x = Pooling(pool_size=2)(x)
    y = Flatten()(x)
    
    f = Dense(filters//2,
              activation='relu',
              kernel_initializer='he_normal')(y)
    
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(f)

    model = Model(inputs=inputs, outputs=outputs)

#     plot_model(model, to_file='dense_net_graph.png')

    return model





num_classes = 38
input_shape = (71,71, 1)





# model = dense_net(input_shape=input_shape, num_classes=num_classes, depth=3, k=4)

model = keras.applications.Xception(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
)

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=opt.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

lr_scheduler = LearningRateScheduler(lr_schedule)

index=np.arange(np.shape(x_train)[0])
np.random.shuffle(index)
# print(index[0:20])

x_train = x_train[index]
y_train = y_train[index]

def data_prep(image):
    # image = np.array(image)
    noise = np.array(np.random.random((71,71,1))*200, dtype='uint8')
    
    if np.mean(image) < 127:
        image = np.clip(image+noise, 0, 255)
    else:
        image = np.clip(image-noise, 0, 255)
    
    image = cv2.GaussianBlur(image,(5,5),0)
    image = np.expand_dims(image, axis=2)
    
    return image

datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=20,
            rotation_range=10,
            horizontal_flip=False,
            vertical_flip=False,
            zoom_range=(0.8, 1.1),
            fill_mode='reflect',
            preprocessing_function=data_prep)


batch_size=64
epochs = 200

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size, subset='training')
test_generator = datagen.flow(x_train, y_train, batch_size=batch_size//4, subset='validation')

history = model.fit(train_generator, steps_per_epoch=32,
                    validation_data=test_generator, validation_steps=32,
                    epochs=epochs, verbose=1,
                    callbacks=[lr_scheduler])

model_json = model.to_json()
with open("model.json", "w") as json_file:
		json_file.write(model_json)
model.save_weights("model.h5")


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()