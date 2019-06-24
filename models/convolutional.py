import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from keras import regularizers
import numpy as np


def leNet(dropout_pool1, dropout_pool2, dropout_dense1, dropout_dense2, l1, l2):
    model = Sequential()

    model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1),
                     kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(AveragePooling2D())
    if dropout_pool1 > 0:
        model.add(Dropout(dropout_pool1))

    model.add(Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(AveragePooling2D())
    if dropout_pool2 > 0:
        model.add(Dropout(dropout_pool2))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_dense1 > 0:
        model.add(Dropout(dropout_dense1))

    model.add(Dense(units=84, activation='relu', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_dense2 > 0:
        model.add(Dropout(dropout_dense2))

    model.add(Dense(units=10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def leNet_model_wide(dropout, l1, l2):
    return leNet(dropout, dropout, dropout, dropout, l1, l2)


def leNet_pooling(dropout, l1, l2):
    return leNet(dropout, dropout, 0, 0, l1, l2)


def leNet_dense(dropout, l1, l2):
    return leNet(0, 0, dropout, dropout, l1, l2)


def vgg(dataset, dropout_pool1, dropout_pool2, dropout_pool3, dropout_dense1, dropout_dense2, l1, l2):
    model = Sequential()
    if dataset == 'cifar':
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3),
                         kernel_regularizer=regularizers.l1_l2(l1, l2)))
    elif dataset == 'mnist':
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1),
                         kernel_regularizer=regularizers.l1_l2(l1, l2)))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pool1 > 0:
        model.add(Dropout(dropout_pool1))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pool2 > 0:
        model.add(Dropout(dropout_pool2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1_l2(l1, l2)))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pool3 > 0:
        model.add(Dropout(dropout_pool3))

    model.add(Flatten())

    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_dense1 > 0:
        model.add(Dropout(dropout_dense1))

    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform',
                    kernel_regularizer=regularizers.l1_l2(l1, l2)))
    if dropout_dense2 > 0:
        model.add(Dropout(dropout_dense2))

    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def vgg_model_wide(dataset, dropout, l1, l2):
    return vgg(dataset, dropout, dropout, dropout, dropout, dropout, l1, l2)


def vgg_pooling(dataset, dropout, l1, l2):
    return vgg(dataset, dropout, dropout, dropout, 0, 0, l1, l2)


def vgg_dense(dataset, dropout, l1, l2):
    return vgg(dataset, 0, 0, 0, dropout, dropout, l1, l2)
