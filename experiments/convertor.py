from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D

from art.utils import load_mnist

import h5py
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

from numpy.random import seed
seed(42)# keras seed fixing
# import tensorflow as tf
# tf.random.set_seed(42)# tensorflow seed fixing

model = Sequential()

model.add(Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D())

model.add(Conv2D(50, (5, 5), activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=500, activation='relu'))

model.add(Dense(units=10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

print(x_test[:10])

# model.fit(x_train, y_train, batch_size=128, epochs=10)

print(model.evaluate(x_test, y_test))

# print(model.layers[5].get_weights()[0])

model.load_weights('leNet5_weights_sparse.h5')


print(model.evaluate(x_test, y_test))


