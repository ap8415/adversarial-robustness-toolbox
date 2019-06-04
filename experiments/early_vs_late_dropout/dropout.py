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

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

print("EARLY DROPOUT TEST:")

for dropout in range(0, 16):
    classifier = neural_networks.six_layer_nn(x_train.shape[1:], dropout_levels[dropout], 0)
    classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on dropout level %.2f%%: %.2f%%" % (dropout_levels[dropout], acc * 100))

    # Craft adversarial samples with CW attack
    adv_crafter = CarliniL2Method(classifier, targeted=False)
    x_test_adv = adv_crafter.generate(x=x_test[:1000])

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on adversarial sample for dropout %.2f%%: %.2f%%" % (dropout_levels[dropout], acc * 100))

    # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
    perturbations = np.absolute((x_test_adv - x_test[:1000]))
    l1_perturbations = [LA.norm(perturbation, 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation, 2) for perturbation in perturbations]
    avg_l1_perturbation = np.average(l1_perturbations)
    avg_l2_perturbation = np.average(l2_perturbations)
    print("\nAverage L1-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.2f%%"
          % (dropout_levels[dropout], avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.2f%%"
          % (dropout_levels[dropout], avg_l2_perturbation))


print("\n LATE DROPOUT TEST:")

for dropout in range(0, 16):
    classifier = neural_networks.six_layer_nn(x_train.shape[1:], 0, dropout_levels[dropout])
    classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on dropout level %.2f%%: %.2f%%" % (dropout_levels[dropout], acc * 100))

    # Craft adversarial samples with CW attack
    adv_crafter = CarliniL2Method(classifier, targeted=False)
    x_test_adv = adv_crafter.generate(x=x_test[:1000])

    # Evaluate the classifier on the adversarial examples
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("\nTest accuracy on adversarial sample for dropout %.2f%%: %.2f%%" % (dropout_levels[dropout], acc * 100))

    # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
    perturbations = np.absolute((x_test_adv - x_test[:1000]))
    l1_perturbations = [LA.norm(perturbation, 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation, 2) for perturbation in perturbations]
    avg_l1_perturbation = np.average(l1_perturbations)
    avg_l2_perturbation = np.average(l2_perturbations)
    print("\nAverage L1-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.2f%%"
          % (dropout_levels[dropout], avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.2f%%"
          % (dropout_levels[dropout], avg_l2_perturbation))


