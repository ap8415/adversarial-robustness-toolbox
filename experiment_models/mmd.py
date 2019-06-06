from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath
from experiment_models.utils import mmd_evaluation

sys.path.append(abspath('.'))

from art.utils import load_mnist_vectorized

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()

from time import time

print(time())

x_1 = x_train[:1000]
x_2 = x_train[1000:2000]

print(mmd_evaluation(x_1, x_2))

print(time())