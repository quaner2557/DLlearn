class getimages:
    def __init__(self, dir, size):
        self.dir = dir
        self.size = size

    def getimages(self):
        import os
        import numpy as np
        import cv2
        matrix = np.empty((self.size, 1, 256, 256), dtype="float32")
        label = np.empty((self.size,), dtype="uint8")
        count = 0
        for j in range(len(self.dir)):
            assert os.path.exists(self.dir[j])
            assert os.path.isdir(self.dir[j])
            imageList = os.listdir(self.dir[j])
            for i in range(len(imageList)):
                img = cv2.imread(self.dir[j] + '/' + imageList[i], 0)
                res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                arr = np.asarray(res, dtype="float32")
                # arr -= np.mean(arr)
                matrix[count, :, :, :] = arr
                label[count] = int(j)
                count += 1
        return matrix, label

import h5py
from keras.models import model_from_json
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import initializers
from keras.layers.normalization import BatchNormalization
import keras.models

# 读取测试数据
X_test, y_test = getimages([r"/media/quan/软件/AAAAA/SAR/test/Bulk",
                          r"/media/quan/软件/AAAAA/SAR/test/Contain",
                          r"/media/quan/软件/AAAAA/SAR/test/oil"], 75).getimages()

y_test = np_utils.to_categorical(y_test, 3)

# 读取model
model = model_from_json(open('ds_model_architecture.json').read())
model.load_weights('ds_model_weights.h5')

rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

score = model.evaluate(X_test, y_test,batch_size=30)
print(score)