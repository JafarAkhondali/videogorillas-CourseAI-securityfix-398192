import numpy as np
from skimage.io import imread, imsave
from scipy.misc import imresize

w = 5000
h = 5000

iw = 64
ih = 64

canvas = np.zeros((h, w, 3), dtype='uint8')

vectors_2d = np.load('vectors_2d.npy')

vectors_rescaled = np.zeros(vectors_2d.shape)

min_y = np.min(vectors_2d[:, 0])
max_y = np.max(vectors_2d[:, 0])

min_x = np.min(vectors_2d[:, 1])
max_x = np.max(vectors_2d[:, 1])

# u \in [a, b] -> [c, d]
# v = (x - a) * (d - c) / (b - a) + c \in [c, d]

vectors_rescaled[:, 0] = (vectors_2d[:, 0] - min_y) * ((h - ih) - ih) / (max_y - min_y) + ih
vectors_rescaled[:, 1] = (vectors_2d[:, 1] - min_x) * ((w - iw) - iw) / (max_x - min_x) + iw

image_filenames = open('filenames.txt').readlines()

for i, fn in enumerate(image_filenames):
    img = imread('dataset/' + fn.strip())
    img = imresize(img, (iw, ih))
    if len(img.shape) == 2:
        continue
    print(fn)
    x = int(vectors_rescaled[i, 1] - iw/2)
    y = int(vectors_rescaled[i, 0] - ih/2)
    canvas[y: y+ih, x: x+iw] = img

imsave('canvas.png', canvas)

print('a')

a = [0, 1, 2, 3, 4, 5, 6, 7]

print(a[2:4])
# [2, 3]
