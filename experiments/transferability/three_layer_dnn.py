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
np.set_printoptions(threshold=sys.maxsize)

def to_one_hot(c):
    """
    Converts c to a one-hot representation.
    :param c: class
    :return: one-hot representation.
    """
    enc = [0.0 for i in range(0, 10)]
    enc[c] = 1.0
    return enc


dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]

for dropout in range(10, 11):
    dropout_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, dropout_levels[dropout], 0, 0)
    baseline_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
    dropout_classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)
    baseline_classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

    true_labels = np.argmax(y_test[:1000], axis=1)
    target_labels = np.array([to_one_hot((c + 1) % 10) for c in true_labels])

    # Craft adversarial samples with CW attack
    # We direct the attacks to find an adversarial sample with class (true label + 1) mod 10.
    baseline_attacker = CarliniL2Method(baseline_classifier, targeted=True, confidence=-20.0)
    dropout_attacker = CarliniL2Method(dropout_classifier, targeted=True, confidence=-20.0)
    x_adv_baseline = baseline_attacker.generate(x=x_test[:1000], y=target_labels)
    x_adv_dropout = dropout_attacker.generate(x=x_test[:1000], y=target_labels)

    # Verify transferability percentage on dropout adversarial examples to baseline model
    baseline_preds = np.argmax(baseline_classifier.predict(x_adv_dropout), axis=1)
    baseline_transfer = (np.sum(baseline_preds == true_labels) / 1000) * 100
    print("\nAccuracy on adversarial samples generated on the dropout model evaluated by baseline model:"
          "%.3f%%" % baseline_transfer)

    # Verify transferability percentage on baseline adversarial examples to dropout model
    dropout_preds = np.argmax(dropout_classifier.predict(x_adv_baseline), axis=1)
    dropout_transfer = (np.sum(dropout_preds == true_labels) / 1000) * 100
    print("\nAccuracy on adversarial samples generated on the baseline model evaluated by dropout model:"
          "%.3f%%" % dropout_transfer)

    predictions_delta = (np.sum(baseline_preds == dropout_preds) / baseline_preds.shape[0]) * 100
    print("Fraction of predictions that are the same on both baseline and dropout: %.3f%%" % predictions_delta)

    print('Verifying some hunches...')

    PROBABILITIES_BASELINE_ADV = baseline_classifier.predict(x_adv_baseline)
    CLASSES_BASELINE_ADV = np.argmax(PROBABILITIES_BASELINE_ADV, axis=1)
    ACC_BASELINE_ADV = (np.sum(CLASSES_BASELINE_ADV == true_labels) / 1000) * 100

    PROBABILITIES_DROPOUT_ADV = dropout_classifier.predict(x_adv_dropout)
    CLASSES_DROPOUT_ADV = np.argmax(PROBABILITIES_DROPOUT_ADV, axis=1)
    ACC_DROPOUT_ADV = (np.sum(CLASSES_DROPOUT_ADV == true_labels) / 1000) * 100


    # print('Baseline probabilities on baseline adversarial : \n\n')
    # print(PROBABILITIES_BASELINE_ADV)
    # print('\n')
    #
    # print('Dropout probabilities on dropout adversarial : \n\n')
    # print(PROBABILITIES_DROPOUT_ADV)
    # print('\n')

    print('Baseline predictions on baseline adversarial : \n\n')
    print(CLASSES_BASELINE_ADV)
    print('\n')

    print('Dropout predictions on dropout adversarial : \n\n')
    print(CLASSES_DROPOUT_ADV)
    print('\n')

    print('Dropout predictions on baseline adversarial are: \n\n')
    print(dropout_preds)
    print('\n')

    print('Baseline predictions on dropout adversarial are: \n\n')
    print(baseline_preds)
    print('\n')

    print('True labels are: \n')
    print(true_labels)

    print('Acc of baseline predictions on baseline adversarial : \n\n')
    print(ACC_BASELINE_ADV)
    print('\n')

    print('Acc of dropout predictions on dropout adversarial : \n\n')
    print(ACC_DROPOUT_ADV)
    print('\n')





