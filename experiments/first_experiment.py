"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import sys
from os.path import abspath

from experiment_models.utils import to_one_hot

sys.path.append(abspath('.'))

import numpy as np

from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.utils import load_mnist_vectorized, get_labels_np_array
from experiment_models import neural_networks

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()

np.set_printoptions(threshold=sys.maxsize)

print(y_test[0])
#
#
# classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
#
#
# classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)
#
# # Evaluate the classifier on the test set
# preds = np.argmax(classifier.predict(x_test), axis=1)
# acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
#
# # Craft adversarial samples with CW attack
#
# attacker = CarliniLInfMethod(classifier,
#                              targeted=True)
#
# x_real = x_test[:1]
# y_real = np.argmax(y_test[:1], axis=1)
# x_test_adv = attacker.generate(x_real, np.array([to_one_hot(1)]))
# print(x_real[0] - x_test_adv[0])
# print('PULA')
# print(classifier.predict(x_real, logits=True))
# print(classifier.predict(x_test_adv, logits=True))
# print(classifier.predict(x_real))
# print(classifier.predict(x_test_adv))






# classifier = neural_networks.two_layer_dnn(x_train.shape[1:], 0, 0, 0)
# classifier.fit(x_train, y_train, nb_epochs=2, batch_size=128)
#
# # Evaluate the classifier on the test set
# preds = classifier.predict(x_test)
#
# x1 = get_labels_np_array(preds)
# print(x1)




