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
from art.utils import load_mnist_vectorized, load_mnist
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation

sns.set()

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment.")
parser.add_argument("attack_type", help="The attack type that we use to perform evaluation.")
parser.add_argument("-binary_steps", help="The number of BS steps used by the attack.", type=int, default=20)
parser.add_argument("-confidence", help="The confidence parameter of the attack.", type=int, default=0)
args = parser.parse_args()

l1_regularization = [0, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]


classifier = None
x_train = None
y_train = None
x_test = None
y_test = None
min_ = None
max_ = None

if args.experiment_type in ["two_layer_dnn", "three_layer_dnn", "five_layer_dnn", "six_layer_dnn"]:
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()
elif args.experiment_type == "simple_cnn":
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
elif args.experiment_type == "leNet5":
    (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
    # Pad images to 32x32 size in order to fit the LeNet architecture
    x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

# We select the examples from the test set which are ones or sevens respectively.
# We have 1135 ones, and 1028 sevens in the test set.
ones = np.where(np.argmax(y_test, axis=1) == 1)[0]
sevens = np.where(np.argmax(y_test, axis=1) == 7)[0]

def to_one_hot(c):
    """
    Converts c to a one-hot representation.
    :param c: class
    :return: one-hot representation.
    """
    enc = [0.0 for i in range(0, 10)]
    enc[c] = 1.0
    return enc


print('\n%s : L1 REG EXPERIMENT - GENERATES HEATMAPS: FROM 0 TO 0.0009 REG LEVEL' % args.experiment_type)

for l1_reg in range(0, 10):
    heatmap = np.zeros((28, 28))
    for i in range(0, 4):

        if args.experiment_type == "two_layer_dnn":
            classifier = neural_networks.two_layer_dnn(x_train.shape[1:], 0, l1_regularization[l1_reg], 0)
        elif args.experiment_type == "three_layer_dnn":
            classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, 0, l1_regularization[l1_reg], 0)
        # elif args.experiment_type == "five_layer_dnn":
        #     classifier = neural_networks.symmetric_five_layer_nn(x_train.shape[1:],
        #                                                          dropout_levels[dropout], dropout_levels[dropout])
        elif args.experiment_type == "six_layer_dnn":
            classifier = neural_networks.asymmetric_six_layer_nn_regularized(x_train.shape[1:], l1_regularization[l1_reg])
        # elif args.experiment_type == "simple_cnn":
        #     classifier = convolutional.simple_cnn(dropout_levels[dropout])
        # elif args.experiment_type == "leNet5":
        #     classifier = convolutional.leNet_cnn_single(dropout_levels[dropout])
        # TODO: add other types of experiments; the only real variable here is the classifier.

        classifier.fit(x_train, y_train, nb_epochs=20, batch_size=128)

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
        print("\nTest accuracy on L1 regularization level %.2f%%: %.3f%%" % (l1_regularization[l1_reg], acc))

        # Craft adversarial samples with CW attack, disguising ones as sevens
        attacker = None
        if args.attack_type == "carlini_l2":
            attacker = CarliniL2Method(classifier,
                                       targeted=True, binary_search_steps=args.binary_steps, confidence=args.confidence)
        elif args.attack_type == "carlini_lInf":
            attacker = CarliniLInfMethod(classifier, targeted=True, confidence=args.confidence)
        x_real = np.take(x_test, ones, axis=0)
        print(x_real[0].shape)
        y_real = np.array([1 for x in x_real])
        y_target = np.array([to_one_hot(7) for _ in x_real])
        x_test_adv = attacker.generate(x_real, y=y_target)

        # Evaluate the classifier on the adversarial examples
        adversarial_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        adversarial_acc = (np.sum(adversarial_preds == y_real) / len(y_real)) * 100
        print("\nTest accuracy on adversarial sample for L1 REGULARIZATION %.2f%%: %.3f%%" %
              (l1_regularization[l1_reg], adversarial_acc))

        # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
        perturbations = np.absolute((x_test_adv - x_real))
        l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
        l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
        lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
        avg_l1_perturbation = np.average(l1_perturbations)
        avg_l2_perturbation = np.average(l2_perturbations)
        avg_lInf_perturbation = np.average(lInf_perturbations)
        print("\nAverage L1-norm perturbation from %s attack for l1 reg %.2f%%: %.4f%%"
              % (args.attack_type, l1_regularization[l1_reg], avg_l1_perturbation))
        print("\nAverage L2-norm perturbation from %s attack for l1 reg %.2f%%: %.4f%%"
              % (args.attack_type, l1_regularization[l1_reg], avg_l2_perturbation))
        print("\nAverage LInf-norm perturbation from %s attack for l1 reg %.2f%%: %.4f%%"
              % (args.attack_type, l1_regularization[l1_reg], avg_lInf_perturbation))

        linear_mmd_real_vs_adversarial = mmd_evaluation(x_real, x_test_adv)
        print('Estimate of Maximum Mean Discrepancy using the normalized linear kernel: %.10f%%'
              % linear_mmd_real_vs_adversarial)
        print('And in log-scale: %.6f%%' % np.math.log(linear_mmd_real_vs_adversarial))

        for perturbation in perturbations:
            heatmap = heatmap + np.reshape(perturbation, (28, 28))

    heatmap = heatmap / (4 * len(ones))

    plt.figure()
    fig = sns.heatmap(heatmap).get_figure()
    fig.savefig(args.experiment_type + '_heatmap_l1_' + str(l1_regularization[l1_reg]) + '_' + args.attack_type + '.png')
