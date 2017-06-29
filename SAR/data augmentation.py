from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import cv2
import numpy as np
import os

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    )

dir1 = r"/media/quan/软件/AAAAA/SAR/BulkCarrier"
dirsave1 =r"/media/quan/软件/AAAAA/SAR/train/BulkCarrier"

dir2 = r"/media/quan/软件/AAAAA/SAR/ContainerShip"
dirsave2 =r"/media/quan/软件/AAAAA/SAR/train/ContainerShip"

dir3 = r"/media/quan/软件/AAAAA/SAR/OilTanker"
dirsave3 =r"/media/quan/软件/AAAAA/SAR/train/OilTanker"

dir = [dir1, dir2, dir3]
dirsave = [dirsave1, dirsave2, dirsave3]

for (d, ds) in zip(dir, dirsave):
    imagelist = os.listdir(d)
    for j in range(len(imagelist)):
        matrix = np.empty((1, 256, 256, 1), dtype='float32')
        img = cv2.imread(d+'/'+imagelist[j], 0)
        res = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        arr = np.asarray(res, dtype="float32")
        matrix[0, :, :, 0] = arr

        i = 0
        for batch in datagen.flow(matrix, batch_size=1, save_to_dir=ds,
                                  save_prefix='B'):
            i += 1
            if i > 60:
                break

