from keras.models import Model
from keras.layers import Input, Dropout
from keras.layers.merge import concatenate
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation


def Unet(nClasses, optimizer=None, input_width=360, input_height=480):

    inputs = Input((input_height, input_width, 3))
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    # conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    # conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)
    m1 = concatenate([up1, conv3])

    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(m1)
    # conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv5)

    up2 = UpSampling2D(size=(2, 2))(conv5)
    m2 = concatenate([up2, conv2])
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(m2)
    # conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv6)

    up3 = UpSampling2D(size=(2, 2))(conv6)
    m3 = concatenate([up3, conv1])
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(m3)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    conv8 = Conv2D(nClasses, (1, 1), activation='sigmoid')(conv7)

    model = Model(inputs=inputs, outputs=conv8)

    if optimizer is not None:
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    m = Unet(2, input_height=288, input_width=352, optimizer='adam')
    m.summary()
    from keras.utils import plot_model
    plot_model(m, show_shapes=True, to_file='Unet.png')
