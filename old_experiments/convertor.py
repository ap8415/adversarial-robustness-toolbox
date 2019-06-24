from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, ReLU, Softmax

from models.utils import load_mnist

import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

from numpy.random import seed
seed(42)# keras seed fixing
# import tensorflow as tf
# tf.random.set_seed(42)# tensorflow seed fixing

model = Sequential()

model.add(Conv2D(20, (5, 5), input_shape=(28, 28, 1), name='conv1'))
model.add(MaxPooling2D())

model.add(Conv2D(50, (5, 5), name='conv2'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(units=500, name='ip1'))
model.add(ReLU())

model.add(Dense(units=10, name='ip2'))
model.add(Softmax())
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

print(x_test[:10])

# model.fit(x_train, y_train, batch_size=128, epochs=10)

print(model.evaluate(x_test, y_test))

# print(model.layers[5].get_weights()[0])

model.load_weights('leNet5_weights_sparse.h5', by_name=True)

print(model.evaluate(x_test, y_test))


