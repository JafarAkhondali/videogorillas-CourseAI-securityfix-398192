import os

import numpy as np
from keras.applications.resnet50 import ResNet50
from scipy.misc import imread, imresize

file_names = os.listdir('dataset/')

model = ResNet50(include_top=False, pooling='avg')

batch_size = 16

for i in range(0, len(file_names), batch_size):
    batch = file_names[i: i + batch_size]
    x_batch = np.zeros((1, 224, 224, 3), dtype='float')

    for j, fn in enumerate(batch):
        img = imread('dataset/' + fn, mode='RGB')
        img = imresize(img, (224, 224, 3))
        x_batch[j] = img

    x_batch = x_batch / 127.5 - 1

    prediction = model.predict(x_batch)

    for j, fn in enumerate(batch):
        np.save('vectors/' + fn + '.npy', prediction[j])
