import sys
from os.path import abspath

sys.path.append(abspath('.'))

from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation, to_one_hot

from foolbox.models import KerasModel
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, AveragePooling2D
from foolbox.batch_attacks import CarliniWagnerL2Attack, RandomPGD
from foolbox.criteria import Misclassification, TargetClass

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
parser.add_argument("attack_type", help="The type of attack used. Can be l2 or linf.")
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

ones = np.where(np.argmax(y_test, axis=1) == 1)[0]

import time

print("Current time: %.2f" % time.time())

dr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]

l1_std = []
l2_std = []
linf_std = []

l1_normalized = []
l2_normalized = []
linf_normalized = []

failure_rate = []

for dropout in dr:
    kmodel = None

    if args.experiment_type == "three_layer_dnn":
        kmodel = neural_networks.three_layer_dnn_foolbox(x_train.shape[1:], 300, 100, dropout, 0, 0)
    elif args.experiment_type == "five_layer_dnn":
        kmodel = neural_networks.symmetric_five_layer_nn_foolbox(x_train.shape[1:], dropout, dropout)
    elif args.experiment_type == "six_layer_dnn":
        kmodel = neural_networks.asymmetric_six_layer_nn_foolbox(x_train.shape[1:], dropout, dropout)
    elif args.experiment_type == "VGG":
        kmodel = convolutional.mini_VGG_foolbox(dropout, dropout, 0, "mnist")
    elif args.experiment_type == "leNet5":
        kmodel = convolutional.leNet_cnn_foolbox(dropout, dropout, "mnist")

    # kmodel.fit(x_train, y_train, epochs=10, batch_size=128)
    kmodel.fit(x_train, y_train, epochs=50, batch_size=128)

    x_sample = np.take(x_test, ones, axis=0)

    # We exclude those examples which are not predicted by the classifier as 1s.
    preds = np.argmax(kmodel.predict(x_sample), axis=1)
    true_ones = np.where(preds == 1)[0]
    x_sample = np.take(x_sample, true_ones, axis=0)

    y_sample = np.array([to_one_hot(1) for _ in x_sample])
    preds = np.argmax(kmodel.predict(x_sample), axis=1)

    attack = None
    if args.attack_type == 'l2':
        attack = CarliniWagnerL2Attack(kmodel, TargetClass(7))
    else:
        attack = RandomPGD(kmodel, TargetClass(7))

    adversarial = None
    if args.attack_type == 'l2':
        adversarial = attack(x_sample, np.argmax(y_sample, axis=1), binary_search_steps=5, max_iterations=600)
    else:
        adversarial = attack(x_sample, np.argmax(y_sample, axis=1), iterations=30)

    # For those samples for which the L2 method does not produce an adversarial sample within the attack parameters,
    # we exclude them from the perturbation evaluation.

    failed = 0
    misclassified = 0

    perturbations = []
    adv_examples = []
    orig_examples = []
    correct_labels = []

    adv_examples_no_misclassification = []
    orig_examples_no_misclassification = []

    orig_examples_failed = []
    correct_labels_failed = []
    for i in range(0, len(adversarial)):
        # If the attack fails, an array of NaN's is returned. We check for this here.
        if not any([np.isnan(x) for x in adversarial[i].flatten()]):
            perturbations.append(np.abs(adversarial[i] - x_sample[i]))
            adv_examples.append(adversarial[i])
            orig_examples.append(x_sample[i])
            correct_labels.append(y_sample[i])

            # We count the misclassified samples as well, as those are returned as adversarials equal to the input
            # image. All misclassified samples will reach this if-branch by definition of the adversarial attack.
            if np.argmax(y_sample[i]) != preds[i]:
                misclassified += 1
            else:
                adv_examples_no_misclassification.append(adversarial[i])
                orig_examples_no_misclassification.append(x_sample[i])
        else:
            failed += 1
            orig_examples_failed.append(x_sample[i])
            correct_labels_failed.append(y_sample[i])

    print('Initially failed: %d' % failed)
    print('Misclassified: (should be 0!):    %d' % misclassified)

    orig_examples_failed = np.array(orig_examples_failed)
    correct_labels_failed = np.array(correct_labels_failed)

    # If the attack failed on any samples, we retry the attack with increased bounds on binary search and
    # number of iterations to ensure success.
    # If on a sample the attack fails even with these parameters, the sample is extremely resilient to attacks, and
    # we stop trying to attack it and instead incorporate it into our metrics.
    if len(orig_examples_failed) > 0:
        adversarial_strong = None
        if args.attack_type == 'l2':
            adversarial_strong = attack(orig_examples_failed, np.argmax(correct_labels_failed, axis=1),
                                        binary_search_steps=15, max_iterations=1000)
        else:
            adversarial_strong = attack(orig_examples_failed, np.argmax(correct_labels_failed, axis=1),
                                        iterations=75)

        for i in range(0, len(adversarial_strong)):
            # If the attack fails, an array of NaN's is returned.
            # Since it failed on the second try, we abandon trying to compute an adversarial sample for this run.
            if not any([np.isnan(x) for x in adversarial_strong[i].flatten()]):
                perturbations.append(np.abs(adversarial_strong[i] - orig_examples_failed[i]))
                adv_examples.append(adversarial_strong[i])
                orig_examples.append(orig_examples_failed[i])
                correct_labels.append(correct_labels_failed[i])

                adv_examples_no_misclassification.append(adversarial_strong[i])
                orig_examples_no_misclassification.append(orig_examples_failed[i])

                failed -= 1

    print('Failed after the 2nd attack attempt: %d' % failed)
    failure_rate.append(failed)

    adv_examples = np.array(adv_examples)
    correct_labels = np.array(correct_labels)
    orig_examples = np.array(orig_examples)
    perturbations = np.array(perturbations)

    l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
    lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]

    l1_std.append(np.average(l1_perturbations))
    l2_std.append(np.average(l2_perturbations))
    linf_std.append(np.average(lInf_perturbations))

    # We normalize by the coefficient of failed attacks to incorporate failure rate into our security measure.
    # For example, if 50% of our attacks fail, we multiply by a coeficient of (1/ (50/100)) = 1/ (1/2) = 2.
    # success_coef = ((10 - failed) / 10)
    # Note: no div-by-0 check as there can't be len(x_sample) failed attacks unless there's a bug in the implementation.
    attack_success_coef = 1 / ((len(x_sample) - failed) / len(x_sample))

    l1_normalized.append(np.average(l1_perturbations) * attack_success_coef)
    l2_normalized.append(np.average(l2_perturbations) * attack_success_coef)
    linf_normalized.append(np.average(lInf_perturbations) * attack_success_coef)

    print("Current time: %.2f" % time.time())

print('Average l1, l2, linf perturbations over the attacks:')
print(l1_std)
print(l2_std)
print(linf_std)

print('Average l1, l2, linf normalized perturbations wrt failed attacks over the successful attacks:')
print(l1_normalized)
print(l2_normalized)
print(linf_normalized)


print('No of failures in attack after 2 attempts:')
print(failure_rate)