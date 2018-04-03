from Unet import Unet
import numpy as np
import os
import random
from scipy.misc import imread, imresize
from keras.callbacks import ModelCheckpoint
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
from scipy.misc import imsave

images_path = '../datasets/portraits/imgs/'
masks_path = '../datasets/portraits/masks/'

img_width = 352
img_height = 288

image_names = [s for s in os.listdir(images_path) if s.endswith('.jpg')]
val_split = int(0.1 * len(image_names))
random.shuffle(image_names)


model = Unet(1, 'adam', input_width=img_width, input_height=img_height)


def f(fn):
    img = imread(fn, mode='RGB')
    img = imresize(img, (img_height, img_width))
    imsave('test.jpg', img)
    return img


pool = ThreadPool(cpu_count())


def generator(images, batch_size=6):
    i = 0
    n = len(images)

    while True:
        # x = np.zeros((batch_size, img_height, img_width, 3), dtype='float32')
        y = np.zeros((batch_size, img_height, img_width, 1), dtype='float32')
        batch = []

        for j in range(batch_size):
            i = (i + 1) % n

            batch.append(images_path + images[i])

            mask = np.load(masks_path + image_names[i].replace('.jpg', '.npy'))
            mask = 255. * mask
            mask = imresize(mask, (img_height, img_width))
            y[j, :, :, 0] = mask / 255.
            # y[j, :, :, 1] = mask

        x = pool.map(f, batch)
        x = np.array(x, dtype='float32')
        x = x / 127.5 - 1

        yield x, y


# model.load_weights('./weights/unet.hdf5')


model.summary()

model.fit_generator(
    generator=generator(image_names[val_split:]),
    validation_data=generator(image_names[:val_split]),
    steps_per_epoch=50,
    validation_steps=10,
    epochs=30000,
    verbose=1,
    callbacks=[
        ModelCheckpoint('weights/unet.hdf5', verbose=1, monitor='val_loss', save_best_only=True)
    ])
