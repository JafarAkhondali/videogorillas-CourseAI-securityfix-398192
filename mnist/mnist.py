from keras import Model
from keras.datasets import mnist
from keras.layers import Dense, Input, Flatten
from keras.utils import plot_model, to_categorical
import numpy as np

# from keras.datasets import fashion_mnist
# from scipy.misc import imsave

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# fashion MNIST: special dataset for Anya
# (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# imsave(str(y_train[0]) + '.png', x_train[0])

print(x_train.shape)
print(y_train.shape)

print(y_train[0:10])

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_train[0:10])


def mnist_model():
    input_layer = Input((28, 28))

    x = Flatten()(input_layer)

    x = Dense(42, activation='sigmoid')(x)
    x = Dense(10, activation='softmax')(x)

    m = Model(inputs=input_layer, outputs=x)

    m.compile(optimizer='adam', loss='categorical_crossentropy')

    return m


model = mnist_model()
model.summary()
plot_model(model, 'model.png', show_shapes=True, show_layer_names=True)

model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_test, y_test),
    batch_size=1024,
    epochs=10)

y_predict = model.predict(x_test, batch_size=1024)

s = '0123456789'
score = 0
for i, p in enumerate(y_predict):
    p_idx = int(np.argmax(p))
    t_idx = int(np.argmax(y_test[i]))
    print('true ' + s[t_idx] + ' predicted ' + s[p_idx])
    score += (p_idx == t_idx)

print('accuracy: %.02f%%' % (100.0 * score / y_test.shape[0]))
