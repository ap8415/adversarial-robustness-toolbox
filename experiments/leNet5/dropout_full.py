"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import numpy as np
import numpy.linalg as LA

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist
from experiment_models import convolutional
from experiment_models.utils import mmd_evaluation_2d

print('\n\n\nLENET5 CNN DROPOUT EXPERIMENT ON ALL LAYERS, DROPOUT FROM 0->0.75 IN INCREMENTS OF 0.05\n\n\n')

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

# Pad images to 32x32 size in order to fit the LeNet architecture
x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

for dropout in range(0, 16):
    classifier = convolutional.leNet_cnn_single(dropout_levels[dropout])
    classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
    print("\nTest accuracy on dropout level %.2f%%: %.3f%%" % (dropout_levels[dropout], acc))

    # Craft adversarial samples with CW attack
    attacker = CarliniL2Method(classifier, targeted=False)
    x_real = x_test[:1000]
    y_real = np.argmax(y_test[:1000], axis=1)
    x_test_adv = attacker.generate(x_real)

    # Evaluate the classifier on the adversarial examples
    adversarial_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    adversarial_acc = (np.sum(adversarial_preds == y_real) / len(y_real)) * 100
    print("\nTest accuracy on adversarial sample for dropout %.2f%%: %.3f%%" %
          (dropout_levels[dropout], adversarial_acc))

    # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
    perturbations = np.absolute((x_test_adv - x_real))
    l1_perturbations = [LA.norm(perturbation.reshape(32 * 32), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.reshape(32 * 32), 2) for perturbation in perturbations]
    avg_l1_perturbation = np.average(l1_perturbations)
    avg_l2_perturbation = np.average(l2_perturbations)
    print("\nAverage L1-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.4f%%"
          % (dropout_levels[dropout], avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from Carlini L2 attack for dropout %.2f%%: %.4f%%"
          % (dropout_levels[dropout], avg_l2_perturbation))

    linear_mmd_real_vs_adversarial = mmd_evaluation_2d(x_test[:1000], x_test_adv)
    print('Estimate of Maximum Mean Discrepancy using the normalized linear kernel: %.10f%%'
          % linear_mmd_real_vs_adversarial)
    print('And in log-scale: %.6f%%' % np.math.log(linear_mmd_real_vs_adversarial))