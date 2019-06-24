import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import regularizers
import numpy as np

from art.classifiers import KerasClassifier
from foolbox.models import KerasModel

# All input values to the network are clipped to fit between 0 and 1.

# BASE KERAS MODELS


def two_layer_dnn(input_shape, dropout, l1_reg, l2_reg):
    model = Sequential()
    model.add(Dense(300, input_shape=input_shape, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Mostly use 300-100 and 500-150 variations as described in MNIST homepage
def three_layer_dnn(input_shape, layer1_size, layer2_size, dropout, l1_reg, l2_reg):

    model = Sequential()
    model.add(Dense(layer1_size,
                    input_shape=input_shape,
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(layer2_size,
                    activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1_reg, l2_reg)))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Symmetric five layer NN.
def symmetric_five_layer_nn(input_shape, early_dropout, late_dropout):
    model = Sequential()
    model.add(Dense(500, input_shape=input_shape, activation='relu'))
    if early_dropout > 0:
        model.add(Dropout(early_dropout))
    model.add(Dense(800, activation='relu'))
    if early_dropout > 0:
        model.add(Dropout(early_dropout))
    model.add(Dense(800, activation='relu'))
    if late_dropout > 0:
        model.add(Dropout(late_dropout))
    model.add(Dense(500, activation='relu'))
    if late_dropout > 0:
        model.add(Dropout(late_dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Symmetric five layer NN.
def symmetric_five_layer_nn_mixed_1(input_shape, dropout):
    model = Sequential()
    model.add(Dense(500, input_shape=input_shape, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(800, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Symmetric five layer NN.
def symmetric_five_layer_nn_mixed_2(input_shape, dropout):
    model = Sequential()
    model.add(Dense(500, input_shape=input_shape, activation='relu'))
    model.add(Dense(800, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(500, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Symmetric five layer NN.
def symmetric_five_layer_nn_l1reg(input_shape, l1reg_early, l1reg_late):
    model = Sequential()
    model.add(Dense(500, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l1(l1reg_early)))
    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1(l1reg_early)))
    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1(l1reg_late)))
    model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1(l1reg_late)))
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def asymmetric_six_layer_nn(input_shape, early_dropout, late_dropout):
    model = Sequential()
    model.add(Dense(1000, input_shape=input_shape, activation='relu'))
    if early_dropout > 0:
        model.add(Dropout(early_dropout))
    model.add(Dense(800, activation='relu'))
    if early_dropout > 0:
        model.add(Dropout(early_dropout))
    model.add(Dense(600, activation='relu'))
    if late_dropout > 0:
        model.add(Dropout(late_dropout))
    model.add(Dense(400, activation='relu'))
    if late_dropout > 0:
        model.add(Dropout(late_dropout))
    model.add(Dense(300, activation='relu'))
    if late_dropout > 0:
        model.add(Dropout(late_dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def asymmetric_six_layer_nn_mixed_1(input_shape, dropout):
    model = Sequential()
    model.add(Dense(1000, input_shape=input_shape, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(800, activation='relu'))
    model.add(Dense(600, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(300, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def asymmetric_six_layer_nn_mixed_2(input_shape, dropout):
    model = Sequential()
    model.add(Dense(1000, input_shape=input_shape, activation='relu'))
    model.add(Dense(800, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(600, activation='relu'))
    model.add(Dense(400, activation='relu'))
    if dropout > 0:
        model.add(Dropout(dropout))
    model.add(Dense(300, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def asymmetric_six_layer_nn_l1reg(input_shape, l1_reg):
    model = Sequential()
    model.add(Dense(1000, input_shape=input_shape, activation='relu',
                    kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(Dense(600, activation='relu', kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l1(l1_reg)))
    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l1(l1_reg)))

    model.add(Dense(10, activation='softmax', kernel_regularizer=regularizers.l1(l1_reg)))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


# Input shape should be 28*28 = 784 for MNIST data
def two_layer_dnn_art(input_shape, dropout, l1_reg, l2_reg):
    model = two_layer_dnn(input_shape=input_shape, dropout=dropout, l1_reg=l1_reg, l2_reg=l2_reg)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


# Mostly use 300-100 and 500-150 variations as described in MNIST homepage
def three_layer_dnn_art(input_shape, layer1_size, layer2_size, dropout, l1_reg, l2_reg):
    model = three_layer_dnn(input_shape, layer1_size, layer2_size, dropout, l1_reg, l2_reg)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


# Symmetric five layer NN.
def symmetric_five_layer_nn_art(input_shape, early_dropout, late_dropout):
    model = symmetric_five_layer_nn(input_shape, early_dropout, late_dropout)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


def asymmetric_six_layer_nn_art(input_shape, early_dropout, late_dropout):
    model = asymmetric_six_layer_nn(input_shape, early_dropout, late_dropout)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


def asymmetric_six_layer_nn_l1reg_art(input_shape, l1_reg):
    model = asymmetric_six_layer_nn_l1reg(input_shape, l1_reg)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier
