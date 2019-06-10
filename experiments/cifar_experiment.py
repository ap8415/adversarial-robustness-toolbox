"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.utils import load_mnist_vectorized, load_mnist, load_cifar10, load_cifar10_vectorized
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation

sns.set()

l1_regularization = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]


classifier = None


(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
print(x_train.shape)


def to_one_hot(c):
    """
    Converts c to a one-hot representation.
    :param c: class
    :return: one-hot representation.
    """
    enc = [0.0 for i in range(0, 10)]
    enc[c] = 1.0
    return enc


print('\n%s : L1 REG EXPERIMENT - GENERATES HEATMAPS: FROM 0 TO 0.0009 REG LEVEL' % 'pula')

for l1_reg in range(6, 7):
    heatmap = np.zeros((28, 28))

    classifier = convolutional.mini_VGG()
    # classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
    # TODO: add other types of experiments; the only real variable here is the classifier.

    classifier.fit(x_train, y_train, nb_epochs=100, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
    print("\nTest accuracy on L1 regularization level %.2f%%: %.3f%%" % (l1_regularization[l1_reg], acc))

