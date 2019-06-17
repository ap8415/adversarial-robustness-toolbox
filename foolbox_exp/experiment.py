import sys
from os.path import abspath

sys.path.append(abspath('.'))

from experiment_models import neural_networks
from experiment_models.utils import mmd_evaluation

from foolbox.models import KerasModel
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, AveragePooling2D
from foolbox.batch_attacks import CarliniWagnerL2Attack
from foolbox.criteria import Misclassification

import argparse
import numpy as np
import numpy.linalg as LA
import logging
import tensorflow as tf

from art.utils import load_mnist, load_mnist_vectorized
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment.")
parser.add_argument("-confidence", help="The confidence parameter of the attack.", type=int, default=0)
args = parser.parse_args()

np.set_printoptions(threshold=sys.maxsize)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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

import time

print("Current time: %.2f" % time.time())

# dr = [0, 0.25, 0.5, 0.65]
dr = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
for dropout in dr:
    kmodel = None

    if args.experiment_type == "three_layer_dnn":
        kmodel = neural_networks.three_layer_dnn_foolbox(x_train.shape[1:], 300, 100, dropout, 0, 0)
    elif args.experiment_type == "five_layer_dnn":
        kmodel = neural_networks.symmetric_five_layer_nn_foolbox(x_train.shape[1:], dropout, dropout)
    elif args.experiment_type == "six_layer_dnn":
        kmodel = neural_networks.asymmetric_six_layer_nn_foolbox(x_train.shape[1:], dropout, dropout)
    # elif args.experiment_type == "VGG":
    #     classifier = convolutional.mini_VGG(dropout_levels[dropout], "mnist")
    # elif args.experiment_type == "leNet5":
    #     classifier = convolutional.leNet_cnn_single(dropout_levels[dropout])

    # kmodel.fit(x_train, y_train, epochs=1, batch_size=128)
    kmodel.fit(x_train, y_train, epochs=20, batch_size=128)

    attack = CarliniWagnerL2Attack(kmodel, Misclassification())

    # adversarial = attack(x_test[:10], np.argmax(y_test[:10], axis=1), binary_search_steps=5, max_iterations=600)
    adversarial = attack(x_test[:1000], np.argmax(y_test[:1000], axis=1), binary_search_steps=5, max_iterations=600)

    # For those samples for which the L2 method does not produce an adversarial sample within the attack parameters,
    # we exclude them from the perturbation evaluation. the None given by the attack with the original input.

    failed = 0
    perturbations = []
    adv_examples = []
    orig_examples = []
    correct_labels = []

    orig_examples_failed = []
    correct_labels_failed = []
    for i in range(0, len(adversarial)):
        # If the attack fails, an array of NaN's is returned. We check for this here.
        if not any([np.isnan(x) for x in adversarial[i].flatten()]):
            perturbations.append(np.abs(adversarial[i] - x_test[i]))
            adv_examples.append(adversarial[i])
            orig_examples.append(x_test[i])
            correct_labels.append(y_test[i])
        else:
            failed += 1
            orig_examples_failed.append(x_test[i])
            correct_labels_failed.append(y_test[i])

    print('Initially failed: %d' % failed)

    orig_examples_failed = np.array(orig_examples_failed)
    correct_labels_failed = np.array(correct_labels_failed)

    # If the attack failed on any samples, we retry the attack with increased bounds on binary search and
    # number of iterations to ensure success.
    # If on a sample the attack fails even with these parameters, the sample is extremely resilient to attacks, and
    # we stop trying to attack it and instead incorporate it into our metrics.
    if len(orig_examples_failed) > 0:
        adversarial_strong = attack(orig_examples_failed, np.argmax(correct_labels_failed, axis=1),
                                    binary_search_steps=15, max_iterations=1000)

        for i in range(0, len(adversarial_strong)):
            # If the attack fails, an array of NaN's is returned.
            # Since it failed on the second try, we abandon trying to compute an adversarial sample for this run.
            if not any([np.isnan(x) for x in adversarial_strong[i].flatten()]):
                perturbations.append(np.abs(adversarial_strong[i] - orig_examples_failed[i]))
                adv_examples.append(adversarial[i])
                orig_examples.append(x_test[i])
                correct_labels.append(correct_labels_failed[i])
                failed -= 1

    print('Failed after the 2nd attack attempt: %d' % failed)

    adv_examples = np.array(adv_examples)
    correct_labels = np.array(correct_labels)
    orig_examples = np.array(orig_examples)
    perturbations = np.array(perturbations)

    print("Failed attacks: %d" %
          np.sum(np.argmax(kmodel.predict(adv_examples), axis=1) == np.argmax(correct_labels, axis=1)))
    print("Size of perturbation vectors: %d" % len(perturbations))

    l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
    lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
    linear_mmd_real_vs_adversarial = mmd_evaluation(orig_examples, adv_examples)

    print('Average l1, l2, linf perturbations over the successful attacks:')
    print(np.average(l1_perturbations))
    print(np.average(l2_perturbations))
    print(np.average(lInf_perturbations))

    # We normalize by the coefficient of failed attacks to incorporate failure rate into our security measure.
    # For example, if 50% of our attacks fail, we multiply by a coeficient of (1/ (50/100)) = 1/ (1/2) = 2.
    success_coef = ((10 - failed) / 10)
    # success_coef = ((1000 - failed) / 1000)
    normalization_factor = 1 / success_coef  # Note: no div-by-0 check as there can't be 1000 failed attacks.

    print('Average l1, l2, linf normalized perturbations over the successful attacks:')
    print(np.average(l1_perturbations) * normalization_factor)
    print(np.average(l2_perturbations) * normalization_factor)
    print(np.average(lInf_perturbations) * normalization_factor)

    print("Current time: %.2f" % time.time())
