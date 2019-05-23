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

print(x_train.shape)

classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100)
classifier.fit(x_train, y_train, nb_epochs=5, batch_size=256)

# Evaluate the classifier on the test set
preds = np.argmax(classifier.predict(x_test), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))

# Craft adversarial samples with CW attack
epsilon = .1  # Maximum perturbation
adv_crafter = CarliniL2Method(classifier, targeted=False)
x_test_adv = adv_crafter.generate(x=x_test[:100])

# Evaluate the classifier on the adversarial examples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy on adversarial sample: %.2f%%" % (acc * 100))

perturbations = (x_test_adv - x_test[:100]) * 255  # de-normalize values
avg_perturbation = np.average(perturbations)
print("\nAverage perturbation from Carlini L2 attack: %.2f%%" % avg_perturbation)
