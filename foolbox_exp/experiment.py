import sys
from os.path import abspath

sys.path.append(abspath('.'))

from foolbox.models import KerasModel
import keras.backend as k
from keras import regularizers
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, AveragePooling2D
from foolbox.batch_attacks import CarliniWagnerL2Attack
from foolbox.criteria import Misclassification
from foolbox.distances import Linf
import numpy.linalg as LA

from art.utils import load_mnist

(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
# Pad images to 32x32 size in order to fit the LeNet/VGG architectures
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# model = Sequential()
#
# model.add(Conv2D(6, (3, 3), activation='relu', input_shape=(32, 32, 1)))
# model.add(AveragePooling2D())
# model.add(Dropout(0.3))
# model.add(Conv2D(16, (3, 3), activation='relu'))
# model.add(AveragePooling2D())
# model.add(Dropout(0.3))
#
# model.add(Flatten())
#
# model.add(Dense(units=120, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(units=84, activation='relu'))
# model.add(Dropout(0.3))
#
# model.add(Dense(units=10, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


import time

print(time.time())

dr = [0, 0.25, 0.5, 0.6, 0.7]

for dropout in dr:

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, verbose=0, epochs=20, batch_size=128)

    kmodel = KerasModel(model=model, bounds=(0, 1))

    attack = CarliniWagnerL2Attack(kmodel, Misclassification())

    adversarial = attack(x_test[:1000], np.argmax(y_test[:1000], axis=1), binary_search_steps=10, max_iterations=200)

    preds = np.argmax(model.predict(adversarial), axis=1)

    acc = (np.sum(preds == np.argmax(y_test[:1000], axis=1)) / len(y_test)) * 100
    print(acc)

    perturbations = np.absolute((adversarial - x_test[:1000]))
    l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
    lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]

    print(np.average(l1_perturbations))
    print(np.average(l2_perturbations))
    print(np.average(lInf_perturbations))

    print(time.time())
