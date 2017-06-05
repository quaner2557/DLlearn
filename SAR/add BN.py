#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 20:00:44 2017

@author: quan
"""

class getimages:
    def __init__(self,dir,size):
        self.dir = dir
        self.size = size

    def getimages(self):
        import os
        import numpy as np
        import cv2
        matrix = np.empty((self.size, 1, 64, 64), dtype="float32")
        label = np.empty((self.size,), dtype="uint8")
        count = 0
        for j in range(len(self.dir)):
            assert os.path.exists(self.dir[j])
            assert os.path.isdir(self.dir[j])
            imageList = os.listdir(self.dir[j])
            for i in range(len(imageList)):
                img = cv2.imread(self.dir[j]+'/'+imageList[i],0)
                res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
                arr = np.asarray(res, dtype="float32")
                # arr -= np.mean(arr)
                matrix[count, :, :, :] = arr
                label[count] = int(j)
                count += 1
        return matrix,label

train, label = getimages([r"/media/quan/软件/AAAAA/SAR/BulkCarrier",r"/media/quan/软件/AAAAA/SAR/ContainerShip",r"/media/quan/软件/AAAAA/SAR/OilTanker"],250).getimages()

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0, random_state=40) #X_train shape(200, 1, 256, 256)

# label为1~3共3个类别，keras要求格式为binary class matrices,转化一下格式
y_train = np_utils.to_categorical(y_train, 3)
# y_test = np_utils.to_categorical(y_test, 3)

#生成一个model
model = Sequential()
model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',input_shape = (1,64,64)))
model.add(Convolution2D(128,(3,3),strides=(2,2),padding='valid',data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))  #64*64

model.add(Convolution2D(256,(2,2),strides=(1,1),padding='same',data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first')) #32*32

model.add(Convolution2D(256,(2,2),strides=(1,1),padding='same',data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first')) #16*16

model.add(Flatten())
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(3))
model.add(Activation('softmax'))

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

print('Training ------------')
model.fit(X_train, y_train, batch_size=20,validation_split=0.4,epochs=200)

# config = model.get_config()

# import h5py
# from keras.models import model_from_json
# json_string = model.to_json()
# open('add NB.json','w').write(json_string)
# model.save_weights('add NB.h5')
