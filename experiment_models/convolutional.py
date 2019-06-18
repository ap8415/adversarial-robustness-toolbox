from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D, regularizers

from art.classifiers import KerasClassifier
from foolbox.models import KerasModel


def simple_cnn(dropout):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
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


def leNet_cnn_single(dropout):
    return leNet_cnn(dropout, dropout, dropout, dropout)


def leNet_cnn(dropout_pooling, dropout_dense, type):
    model = Sequential()

    if type == 'cifar10':
        model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3)))
    else:
        model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())
    if dropout_pooling > 0:
        model.add(Dropout(dropout_pooling))

    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(AveragePooling2D())
    if dropout_pooling > 0:
        model.add(Dropout(dropout_pooling))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))
    if dropout_dense > 0:
        model.add(Dropout(dropout_dense))

    model.add(Dense(units=84, activation='relu'))
    if dropout_dense > 0:
        model.add(Dropout(dropout_dense))

    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def leNet_cnn_l1reg(l1reg, type):
    model = Sequential()

    if type == 'cifar10':
        model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 3),
                         kernel_regularizer=regularizers.l1(l1reg)))
    else:
        model.add(Conv2D(6, (5, 5), activation='relu', input_shape=(32, 32, 1),
                         kernel_regularizer=regularizers.l1(l1reg)))
    model.add(AveragePooling2D())

    model.add(Conv2D(16, (5, 5), activation='relu', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(AveragePooling2D())
    model.add(Flatten())

    model.add(Dense(units=120, activation='relu', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dense(units=84, activation='relu', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def mini_VGG(dropout_pooling, dropout_dense, l1_reg, type):
    model = Sequential()
    if type == 'cifar10':
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    else:  # MNIST
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pooling > 0:
        model.add(Dropout(dropout_pooling))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pooling > 0:
        model.add(Dropout(dropout_pooling))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    if dropout_pooling > 0:
        model.add(Dropout(dropout_pooling))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform'))
    if dropout_dense > 0:
        model.add(Dropout(dropout_dense))
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform'))
    if dropout_dense > 0:
        model.add(Dropout(dropout_dense))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def mini_VGG_l1reg(l1reg, type):
    model = Sequential()
    if type == 'cifar10':
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3),
                         kernel_regularizer=regularizers.l1(l1reg)))
    else:  # MNIST
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1),
                         kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dense(200, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l1(l1reg)))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def mini_VGG_foolbox(dropout_pooling, dropout_dense, l1reg, type):
    """
    Implements a VGG-style architecture. This is the only architecture we use which achieves high enough accuracy
    on CIFAR-10.
    """
    model = mini_VGG(dropout_pooling, dropout_dense, l1reg, type)
    classifier = KerasModel(model=model, bounds=(0, 1))
    return classifier


def mini_VGG_l1reg_foolbox(l1reg, type):
    """
    Implements a VGG-style architecture. This is the only architecture we use which achieves high enough accuracy
    on CIFAR-10.
    """
    model = mini_VGG_l1reg(l1reg, type)
    classifier = KerasModel(model=model, bounds=(0, 1))
    return classifier


def leNet_cnn_foolbox(dropout_pooling, dropout_dense, type):
    model = leNet_cnn(dropout_pooling, dropout_dense, type)
    classifier = KerasModel(model=model, bounds=(0, 1))
    return classifier


def leNet_cnn_l1reg_foolbox(l1reg, type):
    model = leNet_cnn_l1reg(l1reg, type)
    classifier = KerasModel(model=model, bounds=(0, 1))
    return classifier


def mini_VGG_art(dropout, type):
    """
    Implements a VGG-style architecture. This is the only architecture we use which achieves high enough accuracy
    on CIFAR-10.
    """
    model = mini_VGG(dropout, type)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


def mini_VGG_l1reg_art(l1reg, type):
    """
    Implements a VGG-style architecture. This is the only architecture we use which achieves high enough accuracy
    on CIFAR-10.
    """
    model = mini_VGG_l1reg(l1reg, type)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


def leNet_cnn_art(dropout_pooling, dropout_dense, type):
    model = leNet_cnn(dropout_pooling, dropout_dense, type)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier


def leNet_cifar_l1reg_art(l1reg, type):
    model = leNet_cnn_l1reg(l1reg, type)
    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier
