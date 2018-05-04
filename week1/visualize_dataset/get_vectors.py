import os
import numpy as np
from keras.applications.nasnet import NASNetLarge
from imageio import imread
from scipy.misc import imresize

file_names = os.listdir('dataset/')

image_shape = (331, 331, 3)
model = NASNetLarge(input_shape=image_shape, include_top=False, pooling='avg')

batch_size = 16

for i in range(0, len(file_names), batch_size):
    batch = file_names[i: i + batch_size]
    x_batch = np.zeros(((len(batch),) + image_shape), dtype='float')

    for j, fn in enumerate(batch):
        print(fn)
        img = imread('dataset/' + fn, pilmode="RGB")
        img = imresize(img, image_shape)
        x_batch[j] = img

    x_batch = x_batch / 127.5 - 1

    prediction = model.predict(x_batch)

    for j, fn in enumerate(batch):
        np.save('vectors/' + fn + '.npy', prediction[j])
