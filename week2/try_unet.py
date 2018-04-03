from Unet import Unet
from scipy.misc import imread, imsave, imresize
import numpy as np

img_width = 352
img_height = 288

img = imread('../datasets/portraits/imgs/5.jpg')

x = np.zeros((1, 288, 352, 3), dtype='float32')
x[0] = imresize(img, (288, 352))

model = Unet(1, 'adam', input_width=img_width, input_height=img_height)
model.load_weights('weights/unet.hdf5')
y = model.predict(x)

imsave('test.jpg', y[0, :, :, 0] * 255)
