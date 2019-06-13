"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import tensorflow as tf

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized, load_mnist
from experiment_models import neural_networks, convolutional
from experiment_models.utils import to_one_hot
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment.")
args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)

dropout_classifier = None
baseline_classifier = None
x_train = None
y_train = None
x_test = None
y_test = None
min_ = None
max_ = None

if args.experiment_type in ["three_layer_dnn", "five_layer_dnn", "six_layer_dnn"]:
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()
elif args.experiment_type in ["VGG", "leNet5"]:
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
    # Pad images to 32x32 size in order to fit the LeNet/VGG architectures
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# We select the examples from the test set which are ones, to use in the binary experiment.
ones = np.where(np.argmax(y_test, axis=1) == 1)[0]

dropout = 0.5

for confidence in [0, 20]:
    for it in range(0, 5):
        print('20 BINARY SEARCH STEPS, CONFIDENCE VALUE:%.1f%%\n\n\n' % float(confidence))

        if args.experiment_type == "three_layer_dnn":
            dropout_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, dropout, 0, 0)
            baseline_classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
        elif args.experiment_type == "five_layer_dnn":
            dropout_classifier = neural_networks.symmetric_five_layer_nn(x_train.shape[1:], dropout, dropout)
            baseline_classifier = neural_networks.symmetric_five_layer_nn(x_train.shape[1:], 0, 0)
        elif args.experiment_type == "six_layer_dnn":
            dropout_classifier = neural_networks.asymmetric_six_layer_nn(x_train.shape[1:], dropout, dropout)
            baseline_classifier = neural_networks.asymmetric_six_layer_nn(x_train.shape[1:], 0, 0)
        elif args.experiment_type == "VGG":
            dropout_classifier = convolutional.mini_VGG(dropout, "mnist")
            baseline_classifier = convolutional.mini_VGG(0, "mnist")
        elif args.experiment_type == "leNet5":
            dropout_classifier = convolutional.leNet_cnn_single(dropout)
            baseline_classifier = convolutional.leNet_cnn_single(0)

        dropout_classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)
        baseline_classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)

        # Multi-class version: take first 1000 samples, target = (true label + 1) mod 10.

        true_labels = np.argmax(y_test[:1000], axis=1)
        target_labels = np.array([to_one_hot((c + 1) % 10) for c in true_labels])

        baseline_attacker = CarliniL2Method(baseline_classifier,
                                            targeted=True, binary_search_steps=20, confidence=float(confidence))
        dropout_attacker = CarliniL2Method(dropout_classifier,
                                           targeted=True, binary_search_steps=20, confidence=float(confidence))
        x_adv_baseline = baseline_attacker.generate(x=x_test[:1000], y=target_labels)
        x_adv_dropout = dropout_attacker.generate(x=x_test[:1000], y=target_labels)

        # Verify transferability percentage on dropout adversarial examples to baseline model
        baseline_preds = np.argmax(baseline_classifier.predict(x_adv_dropout), axis=1)
        baseline_transfer = (np.sum(baseline_preds == true_labels) / len(true_labels)) * 100
        print("\nAccuracy on adversarial samples generated on the dropout model evaluated by baseline model:"
              "%.3f%%" % baseline_transfer)

        # Verify transferability percentage on baseline adversarial examples to dropout model
        dropout_preds = np.argmax(dropout_classifier.predict(x_adv_baseline), axis=1)
        dropout_transfer = (np.sum(dropout_preds == true_labels) / len(true_labels)) * 100
        print("\nAccuracy on adversarial samples generated on the baseline model evaluated by dropout model:"
              "%.3f%%" % dropout_transfer)

        predictions_delta = (np.sum(baseline_preds == dropout_preds) / baseline_preds.shape[0]) * 100
        print("Fraction of predictions that are the same on both baseline and dropout: %.3f%%" % predictions_delta)

        PROBABILITIES_BASELINE_ADV = baseline_classifier.predict(x_adv_baseline)
        CLASSES_BASELINE_ADV = np.argmax(PROBABILITIES_BASELINE_ADV, axis=1)
        ACC_BASELINE_ADV = (np.sum(CLASSES_BASELINE_ADV == true_labels) / len(true_labels)) * 100

        PROBABILITIES_DROPOUT_ADV = dropout_classifier.predict(x_adv_dropout)
        CLASSES_DROPOUT_ADV = np.argmax(PROBABILITIES_DROPOUT_ADV, axis=1)
        ACC_DROPOUT_ADV = (np.sum(CLASSES_DROPOUT_ADV == true_labels) / len(true_labels)) * 100

        print('Acc of baseline predictions on baseline adversarial : \n\n')
        print('%3f%%' % ACC_BASELINE_ADV)
        print('\n')

        print('Acc of dropout predictions on dropout adversarial : \n\n')
        print('%3f%%' % ACC_DROPOUT_ADV)
        print('\n')

        # Binary version: take all 1's from the test set. and target them to be classified as sevens.

        x_real = np.take(x_test, ones, axis=0)
        true_labels = np.array([1 for x in x_real])
        target_labels = np.array([to_one_hot(7) for _ in x_real])

        baseline_attacker = CarliniL2Method(baseline_classifier,
                                            targeted=True, binary_search_steps=20, confidence=float(confidence))
        dropout_attacker = CarliniL2Method(dropout_classifier,
                                           targeted=True, binary_search_steps=20, confidence=float(confidence))
        x_adv_baseline = baseline_attacker.generate(x=x_real, y=target_labels)
        x_adv_dropout = dropout_attacker.generate(x=x_real, y=target_labels)

        # Verify transferability percentage on dropout adversarial examples to baseline model
        baseline_preds = np.argmax(baseline_classifier.predict(x_adv_dropout), axis=1)
        baseline_transfer = (np.sum(baseline_preds == true_labels) / len(true_labels)) * 100
        print("\nAccuracy on adversarial samples generated on the dropout model evaluated by baseline model:"
              "%.3f%%" % baseline_transfer)

        # Verify transferability percentage on baseline adversarial examples to dropout model
        dropout_preds = np.argmax(dropout_classifier.predict(x_adv_baseline), axis=1)
        dropout_transfer = (np.sum(dropout_preds == true_labels) / len(true_labels)) * 100
        print("\nAccuracy on adversarial samples generated on the baseline model evaluated by dropout model:"
              "%.3f%%" % dropout_transfer)

        predictions_delta = (np.sum(baseline_preds == dropout_preds) / baseline_preds.shape[0]) * 100
        print("Fraction of predictions that are the same on both baseline and dropout: %.3f%%" % predictions_delta)

        PROBABILITIES_BASELINE_ADV = baseline_classifier.predict(x_adv_baseline)
        CLASSES_BASELINE_ADV = np.argmax(PROBABILITIES_BASELINE_ADV, axis=1)
        ACC_BASELINE_ADV = (np.sum(CLASSES_BASELINE_ADV == true_labels) / len(true_labels)) * 100

        PROBABILITIES_DROPOUT_ADV = dropout_classifier.predict(x_adv_dropout)
        CLASSES_DROPOUT_ADV = np.argmax(PROBABILITIES_DROPOUT_ADV, axis=1)
        ACC_DROPOUT_ADV = (np.sum(CLASSES_DROPOUT_ADV == true_labels) / len(true_labels)) * 100

        print('Acc of baseline predictions on baseline adversarial : \n\n')
        print('%3f%%' % ACC_BASELINE_ADV)
        print('\n')

        print('Acc of dropout predictions on dropout adversarial : \n\n')
        print('%3f%%' % ACC_DROPOUT_ADV)
        print('\n')
