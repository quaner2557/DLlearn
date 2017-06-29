def getimages(dir,size):
    import os
    import numpy as np
    import cv2
    matrix = np.empty((size, 1, 64, 64), dtype="float32")
    label = np.empty((size,), dtype="uint8")
    count = 0
    for j in range(len(dir)):
        assert os.path.exists(dir[j])
        assert os.path.isdir(dir[j])
        imageList = os.listdir(dir[j])
        for i in range(len(imageList)):
            img = cv2.imread(dir[j] + '/' + imageList[i], 0)
            res = cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)
            arr = np.asarray(res, dtype="float32")
            matrix[count, :, :, :] = arr
            label[count] = int(j)
            count += 1
    return matrix,label

from sklearn.cross_validation import train_test_split

train, label = getimages([r"/media/quan/软件/AAAAA/SAR/BulkCarrier",r"/media/quan/软件/AAAAA/SAR/ContainerShip",r"/media/quan/软件/AAAAA/SAR/OilTanker"],250)
X_train, X_test, y_train, y_test = train_test_split(train, label, test_size=0.2, random_state=42) #X_train shape(200, 1, 256, 256)


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad,RMSprop
from keras.utils import np_utils, generic_utils
from keras.layers.normalization import BatchNormalization

train_datagen = ImageDataGenerator(
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                vertical_flip=True,
                data_format='channels_first',
                fill_mode='reflect')

test_datagen = ImageDataGenerator(rescale=1./255, data_format='channels_first')

# label为1~3共3个类别，keras要求格式为binary class matrices,转化一下格式
y_train = np_utils.to_categorical(y_train, 3)
y_test = np_utils.to_categorical(y_test, 3)

train_generator = train_datagen.flow( X_train,y_train,save_to_dir ='/media/quan/软件/AAAAA/SAR/dir/train',)

validation_generator = test_datagen.flow(X_test,y_test,save_to_dir ='/media/quan/软件/AAAAA/SAR/dir/test')

#生成一个model
model = Sequential()
model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones',input_shape = (1,64,64)))
model.add(Convolution2D(128, (3,3), strides=(2,2),data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))   #37*37

model.add(BatchNormalization(axis=1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones'))
model.add(Convolution2D(256, (2,2), strides=(1,1),data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))  #18*18

model.add(Convolution2D(256, (2,2), strides=(1,1),padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),data_format='channels_first'))  #9*9

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

# Another way to define your optimizer
rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# We add metrics to get more results you want to see
model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit_generator(
        train_generator,
        steps_per_epoch=20,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=20)
