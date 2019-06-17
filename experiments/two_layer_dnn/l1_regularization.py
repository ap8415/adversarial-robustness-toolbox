"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import numpy as np
import numpy.linalg as LA

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized
from experiment_models import neural_networks

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()

print(x_train.shape)

l1_reg_levels = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]

for l1_level in range(0, 5):
    classifier = neural_networks.two_layer_dnn_art(x_train.shape[1:], 0, l1_reg_levels[l1_level], 0)
    classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on L1 regularization level %.5f%%: %.2f%%" % (l1_reg_levels[l1_level], acc * 100))

    # Craft adversarial samples with CW attack
    adv_crafter = CarliniL2Method(classifier, targeted=False)
    x_test_adv = adv_crafter.generate(x=x_test[:1000])

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on adversarial sample for L1 regularization level %.5f%%: %.2f%%"
          % (l1_reg_levels[l1_level], acc * 100))

    # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
    perturbations = np.absolute((x_test_adv - x_test[:1000]))
    l1_perturbations = [LA.norm(perturbation, 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation, 2) for perturbation in perturbations]
    avg_l1_perturbation = np.average(l1_perturbations)
    avg_l2_perturbation = np.average(l2_perturbations)
    print("\nAverage L1-norm perturbation from Carlini L2 attack for L1 regularization level %.5f%%: %.2f%%"
          % (l1_reg_levels[l1_level], avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from Carlini L2 attack for L1 regularization level %.5f%%: %.2f%%"
          % (l1_reg_levels[l1_level], avg_l2_perturbation))

