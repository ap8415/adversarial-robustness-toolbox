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
print(x_train.shape)
print(np.argmax(y_test, axis=1))
print(y_test.shape)
y_second = np.argmax(y_test[:100], axis=1)
y_second[99] = 11
acc = np.sum(y_second == np.argmax(y_test, axis=1)[:100])
print(acc)


true_labels = np.argmax(y_test[:1000], axis=1)
target_labels = np.array([(c + 1) % 10 for c in true_labels])
print(target_labels)


classifier = neural_networks.two_layer_dnn(x_train.shape[1:], 0, 0, 0)
classifier.fit(x_train, y_train, nb_epochs=2, batch_size=128)

# Evaluate the classifier on the test set
preds = classifier.predict(x_test)

x1 = get_labels_np_array(preds)
print(x1)




