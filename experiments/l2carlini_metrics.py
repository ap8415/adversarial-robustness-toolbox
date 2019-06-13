"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import numpy.linalg as LA
import tensorflow as tf
import matplotlib.pyplot as plt

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized, load_mnist
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation
from statistics import mean
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment.")
parser.add_argument("-binary_steps", help="The number of BS steps used by the attack.", type=int, default=20)
parser.add_argument("-confidence", help="The confidence parameter of the attack.", type=int, default=0)
args = parser.parse_args()

# Map arg names to actual model names
actual_names = {'two_layer_dnn': '2-layer DNN',
                'three_layer_dnn': '3-layer DNN',
                'five_layer_dnn': '5-layer DNN',
                'six_layer_dnn': '6-layer DNN',
                'leNet5': 'LeNet5 CNN',
                'VGG': 'Mini-VGG CNN'}

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

classifier = None
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

min_l1_perturbation = []
min_l2_perturbation = []
min_lInf_perturbation = []
min_mmd = []

max_l1_perturbation = []
max_l2_perturbation = []
max_lInf_perturbation = []
max_mmd = []

mean_l1_perturbation = []
mean_l2_perturbation = []
mean_lInf_perturbation = []
mean_mmd = []

for dropout in range(0, 18):
    avg_l1_perturbation = []
    avg_l2_perturbation = []
    avg_lInf_perturbation = []
    mmd_statistic = []
    accuracy = []
    for i in range(0, 10):
        if args.experiment_type == "three_layer_dnn":
            classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, dropout_levels[dropout], 0, 0)
        elif args.experiment_type == "five_layer_dnn":
            classifier = neural_networks.symmetric_five_layer_nn(x_train.shape[1:],
                                                                 dropout_levels[dropout], dropout_levels[dropout])
        elif args.experiment_type == "six_layer_dnn":
            classifier = neural_networks.asymmetric_six_layer_nn(x_train.shape[1:],
                                                                 dropout_levels[dropout], dropout_levels[dropout])
        elif args.experiment_type == "VGG":
            classifier = convolutional.mini_VGG(dropout_levels[dropout], "mnist")
        elif args.experiment_type == "leNet5":
            classifier = convolutional.leNet_cnn_single(dropout_levels[dropout])

        classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
        accuracy.append(acc)

        # Craft adversarial samples with CW attack

        attacker = CarliniL2Method(classifier,
                                   targeted=False, binary_search_steps=args.binary_steps, confidence=args.confidence)

        x_real = x_test[:1000]
        y_real = np.argmax(y_test[:1000], axis=1)
        x_test_adv = attacker.generate(x_real)

        # Calculate the average perturbation and MMD metric. Note that I don't de-normalize the values.
        perturbations = np.absolute((x_test_adv - x_real))
        l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
        l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
        lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
        linear_mmd_real_vs_adversarial = mmd_evaluation(x_test[:1000], x_test_adv)

        avg_l1_perturbation.append(np.average(l1_perturbations))
        avg_l2_perturbation.append(np.average(l2_perturbations))
        avg_lInf_perturbation.append(np.average(lInf_perturbations))
        mmd_statistic.append(linear_mmd_real_vs_adversarial)

    min_l1_perturbation.append(min(avg_l1_perturbation))
    min_l2_perturbation.append(min(avg_l2_perturbation))
    min_lInf_perturbation.append(min(avg_lInf_perturbation))
    min_mmd.append(np.math.log(min(mmd_statistic)))

    max_l1_perturbation.append(max(avg_l1_perturbation))
    max_l2_perturbation.append(max(avg_l2_perturbation))
    max_lInf_perturbation.append(max(avg_lInf_perturbation))
    max_mmd.append(np.math.log(max(mmd_statistic)))

    mean_l1_perturbation.append(mean(avg_l1_perturbation))
    mean_l2_perturbation.append(mean(avg_l2_perturbation))
    mean_lInf_perturbation.append(mean(avg_lInf_perturbation))
    mean_mmd.append(np.math.log(mean(mmd_statistic)))

dropout_levels = [100 * d for d in dropout_levels]  # show levels as percentages

# Plot all the measurements

fig = plt.figure()
plt.plot(dropout_levels, min_l1_perturbation,
         dropout_levels, mean_l1_perturbation,
         dropout_levels, max_l1_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, L1 distance')
plt.xlabel('Dropout%')
plt.ylabel('L1 distance')
plt.savefig(args.experiment_type + '_l1.png')

fig = plt.figure()
plt.plot(dropout_levels, min_l2_perturbation,
         dropout_levels, mean_l2_perturbation,
         dropout_levels, max_l2_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, L2 distance')
plt.xlabel('Dropout%')
plt.ylabel('L2 distance')
plt.savefig(args.experiment_type + '_l2.png')

fig = plt.figure()
plt.plot(dropout_levels, min_lInf_perturbation,
         dropout_levels, mean_lInf_perturbation,
         dropout_levels, max_lInf_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, LInf distance')
plt.xlabel('Dropout%')
plt.ylabel('LInf distance')
plt.savefig(args.experiment_type + '_lInf.png')

# Note: MMD is plotted in log-space for relevancy
fig = plt.figure()
plt.plot(dropout_levels, min_mmd,
         dropout_levels, mean_mmd,
         dropout_levels, max_mmd)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, linear MMD in log-space')
plt.xlabel('Dropout%')
plt.ylabel('Linear MMD')
plt.savefig(args.experiment_type + '_mmd.png')