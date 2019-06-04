"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import numpy as np

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized
from experiment_models import neural_networks

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()

np.set_printoptions(threshold=sys.maxsize)
print(x_train.shape)
print(np.argmax(y_test, axis=1))
print(y_test.shape)
y_second = np.argmax(y_test[:100], axis=1)
y_second[99] = 11
acc = np.sum(y_second == np.argmax(y_test, axis=1)[:100])
print(acc)