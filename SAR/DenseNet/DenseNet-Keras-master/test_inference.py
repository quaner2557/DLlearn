"""Test ImageNet pretrained DenseNet"""

class getimages:
    def __init__(self, dir, size):
        self.dir = dir
        self.size = size

    def getimages(self):
        import os
        import numpy as np
        import cv2
        matrix = np.empty((self.size, 256, 256, 1), dtype="float32")
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
                arr -= np.mean(arr)
                arr /= np.std(arr)
                matrix[count, :, :, 0] = arr
                label[count] = int(j)
                count += 1
        return matrix, label


from keras.models import model_from_json
from keras.utils import np_utils
from keras.optimizers import SGD
from densenet_xiagai import DenseNet
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from sklearn.cross_validation import train_test_split
import numpy as np
from keras.callbacks import ModelCheckpoint
import h5py

# train data
train, label = getimages([r"/media/quan/软件/AAAAA/SAR/train/BulkCarrier",
                          r"/media/quan/软件/AAAAA/SAR/train/ContainerShip",
                          r"/media/quan/软件/AAAAA/SAR/train/OilTanker"], 12512).getimages()


X_train, X_eval, y_train, y_eval = train_test_split(train, label, test_size=0.2, random_state=40)

# label为1~3共3个类别，keras要求格式为binary class matrices,转化一下格式
y_train = np_utils.to_categorical(y_train, 3)
y_eval = np_utils.to_categorical(y_eval, 3)

# 读取测试数据
X_test, y_test = getimages([r"/media/quan/软件/AAAAA/SAR/test/Bulk",
                          r"/media/quan/软件/AAAAA/SAR/test/Contain",
                          r"/media/quan/软件/AAAAA/SAR/test/oil"], 75).getimages()

y_test = np_utils.to_categorical(y_test, 3)

# model
model = DenseNet(nb_dense_block=3, reduction=0.5, classes=3, weights_path=None)

sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

json_string = model.to_json()
open('my_model1.json','w').write(json_string)
# checkpointer = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True, period=1)
# model.fit(X_train, y_train, batch_size=20,  validation_data=(X_eval, y_eval), epochs=20, callbacks=[checkpointer])
#
# model.summary()
#
# score = model.evaluate(X_test, y_test, batch_size=20)
# print(score)
