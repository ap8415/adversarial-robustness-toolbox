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

for dropout in range(0, 16):
    dropout_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, dropout_levels[dropout], 0, 0)
    baseline_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
    dropout_classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)
    baseline_classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    # Craft adversarial samples with CW attack
    # We direct the attacks to find an adversarial sample with class (true label + 1) mod 10.
    baseline_attacker = CarliniL2Method(baseline_classifier, targeted=True)
    dropout_attacker = CarliniL2Method(dropout_classifier, targeted=True)
    x_adv_baseline = baseline_attacker.generate(x=x_test[:1000], y=np.array([(c + 1) % 10 for c in y_test[:1000]]))
    x_adv_dropout = dropout_attacker.generate(x=x_test[:1000], y=np.array([(c + 1) % 10 for c in y_test[:1000]]))

    # Verify transferability percentage on dropout adversarial examples to baseline model
    baseline_preds = np.argmax(baseline_classifier.predict(x_adv_dropout), axis=1)
    baseline_transfer = (np.sum(baseline_preds == np.argmax(y_test, axis=1)) / y_test.shape[0]) * 100
    print("\nAccuracy on adversarial samples generated on the dropout model evaluated by baseline model:"
          "%.3f%%" % baseline_transfer)

    # Verify transferability percentage on baseline adversarial examples to dropout model
    dropout_preds = np.argmax(baseline_classifier.predict(x_adv_baseline), axis=1)
    dropout_transfer = (np.sum(dropout_preds == np.argmax(y_test, axis=1)) / y_test.shape[0]) * 100
    print("\nAccuracy on adversarial samples generated on the baseline model evaluated by dropout model:"
          "%.3f%%" % dropout_transfer)

    predictions_delta = (np.sum(baseline_preds == dropout_preds) / baseline_preds.shape[0]) * 100
    print("Fraction of predictions that are the same on both baseline and dropout: %.3f%%" % predictions_delta)
