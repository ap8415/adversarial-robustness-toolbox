import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
import numpy as np

from art.classifiers import KerasClassifier

# All input values to the network are clipped to fit between 0 and 1.


# Input shape should be 28*28 = 784 for MNIST data
def two_layer_dnn(input_shape):
    model = Sequential()
    model.add(Dense(300, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


# Mostly use 300-100 and 500-150 variations as described in MNIST homepage
def three_layer_dnn(input_shape, layer1_size, layer2_size):
    print(input_shape)

    model = Sequential()
    model.add(Dense(layer1_size, input_shape=input_shape, activation='relu'))
    model.add(Dense(layer2_size, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


# Six-layer NN as described in Ciresan et al, 2011
# note: might drop this if it doesnt work even on the gpus
def six_layer_nn(input_shape):
    model = Sequential()
    model.add(Dense(input_shape, input_shape=(input_shape,), activation='relu'))
    model.add(Dense(2500, activation='relu'))
    model.add(Dense(2000, activation='relu'))
    model.add(Dense(1500, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier
