"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import seaborn as sns

from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.utils import load_mnist_vectorized, load_mnist, load_cifar10, load_cifar10_vectorized
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation

sns.set()

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("attack_type", help="The attack type that we use to perform evaluation.")
parser.add_argument("-binary_steps", help="The number of BS steps used by the attack.", type=int, default=20)
parser.add_argument("-confidence", help="The confidence parameter of the attack.", type=int, default=0)
args = parser.parse_args()

l1_regularization = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]


classifier = None


(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
print(x_train.shape)


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

print('\n%s : L1 REG EXPERIMENT - GENERATES HEATMAPS: FROM 0 TO 0.0009 REG LEVEL' % 'pula')

for dropout in dropout_levels:
    heatmap = np.zeros((28, 28))

    print("DROPOUT LEVEL %.03f%%" % dropout)

    classifier = convolutional.mini_VGG(dropout)
    # classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, 0, 0)
    # TODO: add other types of experiments; the only real variable here is the classifier.

    classifier.fit(x_train, y_train, nb_epochs=1, batch_size=128)

    # Evaluate the classifier on the test set
    preds = np.argmax(classifier.predict(x_test), axis=1)
    acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
    print("\nTest accuracy on dropout level %.2f%%: %.3f%%" % (dropout, acc))

    # Craft adversarial samples with CW attack
    attacker = None
    if args.attack_type == "carlini_l2":
        attacker = CarliniL2Method(classifier,
                                   targeted=False, binary_search_steps=args.binary_steps, confidence=args.confidence)
    elif args.attack_type == "carlini_lInf":
        attacker = CarliniLInfMethod(classifier, targeted=False, confidence=args.confidence)
    x_real = x_test[:1000]
    y_real = np.argmax(y_test[:1000], axis=1)
    x_test_adv = attacker.generate(x_real)

    # Evaluate the classifier on the adversarial examples
    adversarial_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    adversarial_acc = (np.sum(adversarial_preds == y_real) / len(y_real)) * 100
    print("\nTest accuracy on adversarial sample for dropout %.2f%%: %.3f%%" %
          (dropout, adversarial_acc))

    # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
    perturbations = np.absolute((x_test_adv - x_real))
    l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
    lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
    avg_l1_perturbation = np.average(l1_perturbations)
    avg_l2_perturbation = np.average(l2_perturbations)
    avg_lInf_perturbation = np.average(lInf_perturbations)
    print("\nAverage L1-norm perturbation from %s attack for dropout %.2f%%: %.4f%%"
          % (args.attack_type, dropout, avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from %s attack for dropout %.2f%%: %.4f%%"
          % (args.attack_type, dropout, avg_l2_perturbation))
    print("\nAverage LInf-norm perturbation from %s attack for dropout %.2f%%: %.4f%%"
          % (args.attack_type, dropout, avg_lInf_perturbation))

    linear_mmd_real_vs_adversarial = mmd_evaluation(x_test[:1000], x_test_adv)
    print('Estimate of Maximum Mean Discrepancy using the normalized linear kernel: %.10f%%'
          % linear_mmd_real_vs_adversarial)
    print('And in log-scale: %.6f%%' % np.math.log(linear_mmd_real_vs_adversarial))

