from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D

from art.classifiers import KerasClassifier


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


def leNet_cnn(dropout_pool1, dropout_pool2, dropout_fc1, dropout_fc2):
    model = Sequential()

    model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(AveragePooling2D())
    if dropout_pool1 > 0:
        model.add(Dropout(dropout_pool1))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(AveragePooling2D())
    if dropout_pool2 > 0:
        model.add(Dropout(dropout_pool2))

    model.add(Flatten())

    model.add(Dense(units=120, activation='relu'))
    if dropout_fc1 > 0:
        model.add(Dropout(dropout_fc1))

    model.add(Dense(units=84, activation='relu'))
    if dropout_fc2 > 0:
        model.add(Dropout(dropout_fc2))

    model.add(Dense(units=10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    classifier = KerasClassifier(clip_values=(0., 1.), model=model)
    return classifier

def leNet_cnn(dropout):
    return leNet_cnn(dropout, dropout, dropout, dropout)