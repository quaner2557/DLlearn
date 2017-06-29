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
                img = cv2.imread(self.dir[j]+'/'+imageList[i],0)
                res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
                arr = np.asarray(res, dtype="float32")
                # arr -= np.mean(arr)
                matrix[count, :, :, :] = arr
                label[count] = int(j)
                count += 1
        return matrix, label

train, label = getimages([r"/media/quan/软件/AAAAA/SAR/data/BulkCarrier", r"/media/quan/软件/AAAAA/SAR/data/ContainerShip", r"/media/quan/软件/AAAAA/SAR/data/OilTanker"], 250).getimages()

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
from keras import initializers
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=40) #X_train shape(200, 1, 256, 256)

# label为1~3共3个类别，keras要求格式为binary class matrices,转化一下格式
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)

#设置随机数种子,保证实验可重复
import numpy as np
np.random.seed(0)

#生成一个model
model = Sequential()
model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',input_shape = (1,256,256)))
model.add(Convolution2D(32,(3,3), strides=(2,2), padding='valid', data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))  #64*64

model.add(Convolution2D(64,(2,2),strides=(1,1),padding='same',data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first')) #32*32

model.add(Convolution2D(128,(2,2),strides=(1,1),padding='same',data_format='channels_first'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first')) #16*16

model.add(Flatten())
model.add(Dense(50, use_bias=True, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
model.add(Activation('relu'))
model.add(Dropout(0.6))
model.add(Dense(50, use_bias=True, kernel_initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None)))
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

tbCallBack = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

print('Training ------------')
model.fit(X_train, y_train, batch_size=20, validation_split=0.2, epochs=110, callbacks=[tbCallBack])

# print('\nTesting ------------')
# # Evaluate the model with the metrics we defined earlier
# loss, accuracy = model.evaluate(X_test, y_test)
#
# print('test loss: ', loss)
# print('test accuracy: ', accuracy)
config = model.get_config()

import h5py
from keras.models import model_from_json
json_string = model.to_json()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')

# #读取model  
#model = model_from_json(open('my_model_architecture.json').read())  
#model.load_weights('my_model_weights.h5')  

# 绘制结构图
from keras.utils import plot_model
plot_model(model, to_file='model.png')