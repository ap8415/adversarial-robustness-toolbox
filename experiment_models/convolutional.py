import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import regularizers
import numpy as np

from art.classifiers import KerasClassifier

def simple_cnn(dropout):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier

def leNet_cnn(dropout):
    return


