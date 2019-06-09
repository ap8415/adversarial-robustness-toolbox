"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import numpy as np

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized, get_labels_np_array
from experiment_models import neural_networks

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()

np.set_printoptions(threshold=sys.maxsize)


ones = np.where(np.argmax(y_test, axis=1) == 1)
sevens = np.where(np.argmax(y_test, axis=1) == 7)


print('Ones is   ')
print(ones[0].shape)
print('Sevens is...  ')
print(sevens[0].shape)











# classifier = neural_networks.two_layer_dnn(x_train.shape[1:], 0, 0, 0)
# classifier.fit(x_train, y_train, nb_epochs=2, batch_size=128)
#
# # Evaluate the classifier on the test set
# preds = classifier.predict(x_test)
#
# x1 = get_labels_np_array(preds)
# print(x1)




