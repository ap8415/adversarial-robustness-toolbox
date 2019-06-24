import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import regularizers
import numpy as np


def three_layer_dnn(input_shape, output_shape, dropout, l1, l2):
    model = Sequential()

    model.add(Dense(300, input_shape=input_shape, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(100, activation='relu',
                    kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout > 0:
        model.add(Dropout(dropout))

    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def five_layer_dnn(input_shape, output_shape, dropout_1, dropout_2, dropout_3, dropout_4, l1, l2):
    model = Sequential()

    model.add(Dense(500, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_1 > 0:
        model.add(Dropout(dropout_1))

    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_2 > 0:
        model.add(Dropout(dropout_2))

    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_3 > 0:
        model.add(Dropout(dropout_3))

    model.add(Dense(500, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_4 > 0:
        model.add(Dropout(dropout_4))

    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def five_layer_dnn_model_wide(input_shape, output_shape, dropout, l1, l2):
    return five_layer_dnn(input_shape, output_shape, dropout, dropout, dropout, dropout, l1, l2)


def six_layer_dnn(input_shape, output_shape, dropout_1, dropout_2, dropout_3, dropout_4, dropout_5, l1, l2):
    model = Sequential()

    model.add(Dense(1000, input_shape=input_shape, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_1 > 0:
        model.add(Dropout(dropout_1))

    model.add(Dense(800, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_2 > 0:
        model.add(Dropout(dropout_2))

    model.add(Dense(600, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_3 > 0:
        model.add(Dropout(dropout_3))

    model.add(Dense(400, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_4 > 0:
        model.add(Dropout(dropout_4))

    model.add(Dense(300, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_5 > 0:
        model.add(Dropout(dropout_5))

    model.add(Dense(output_shape, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def six_layer_dnn_model_wide(input_shape, output_shape, dropout, l1, l2):
    return six_layer_dnn(input_shape, output_shape, dropout, dropout, dropout, dropout, dropout, l1, l2)
