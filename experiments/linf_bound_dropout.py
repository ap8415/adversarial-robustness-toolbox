"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import numpy.linalg as LA
import logging
import sys
import tensorflow as tf

from art.attacks.carlini import CarliniL2Method, CarliniLInfMethod
from art.utils import load_mnist_vectorized, load_mnist
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment.")
parser.add_argument("dropout", help="Level of dropout for which to perform experiment. Value between 0 and 100.", type=int)
parser.add_argument("-binary_steps", help="The number of BS steps used by the attack.", type=int, default=20)
parser.add_argument("-confidence", help="The confidence parameter of the attack.", type=int, default=0)
args = parser.parse_args()

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


eps_values = [0.3, 0.276, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05]

dropout = float(args.dropout) / 100

print('\n%s : L-INF DROPOUT EXPERIMENT, WITH DROPOUT VALUE %.02f%% \n' % (args.experiment_type, dropout))

for eps in eps_values:
    overall_attack_success = 0
    avg_l1_perturbation = 0
    avg_l2_perturbation = 0
    avg_lInf_perturbation = 0
    print("Running experiment for eps value %.3f " % eps)
    for i in range(0, 5):
        print("Iteration %d of 5:" % (i+1))
        if args.experiment_type == "two_layer_dnn":
            classifier = neural_networks.two_layer_dnn(x_train.shape[1:], dropout, 0, 0)
        elif args.experiment_type == "three_layer_dnn":
            classifier = neural_networks.three_layer_dnn(x_train.shape[1:], 300, 100, dropout, 0, 0)
        elif args.experiment_type == "five_layer_dnn":
            classifier = neural_networks.symmetric_five_layer_nn(x_train.shape[1:],
                                                                 dropout, dropout)
        elif args.experiment_type == "six_layer_dnn":
            classifier = neural_networks.asymmetric_six_layer_nn(x_train.shape[1:],
                                                                 dropout, dropout)
        elif args.experiment_type == "simple_cnn":
            classifier = convolutional.simple_cnn(dropout)
        elif args.experiment_type == "leNet5":
            classifier = convolutional.leNet_cnn_single(dropout)
        # TODO: add other types of experiments; the only real variable here is the classifier.

        classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

        # Evaluate the classifier on the test set
        preds = np.argmax(classifier.predict(x_test), axis=1)
        acc = (np.sum(preds == np.argmax(y_test, axis=1)) / len(y_test)) * 100
        print("\nTest accuracy on dropout level %.2f%%: %.3f%%" % (dropout, acc))

        # Craft adversarial samples with CW attack
        attacker = CarliniLInfMethod(classifier, targeted=False, confidence=args.confidence, eps=eps)
        x_real = x_test[:1000]
        y_real = np.argmax(y_test[:1000], axis=1)
        x_test_adv = attacker.generate(x_real)

        # Evaluate the classifier on the adversarial examples
        adversarial_preds = np.argmax(classifier.predict(x_test_adv), axis=1)
        adversarial_acc = (np.sum(adversarial_preds == y_real) / len(y_real)) * 100
        print("\nTest accuracy on adversarial sample for dropout %.2f%%: %.3f%%" %
              (dropout, adversarial_acc))
        overall_attack_success += adversarial_acc

        # Calculate the average perturbation in L1 and L2 norms. Note that I don't de-normalize the values.
        perturbations = np.absolute((x_test_adv - x_real))
        l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
        l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
        lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
        avg_l1_perturbation += np.average(l1_perturbations)
        avg_l2_perturbation += np.average(l2_perturbations)
        avg_lInf_perturbation += np.average(lInf_perturbations)

        linear_mmd_real_vs_adversarial = mmd_evaluation(x_test[:1000], x_test_adv)
        print('Estimate of Maximum Mean Discrepancy using the normalized linear kernel: %.10f%%'
              % linear_mmd_real_vs_adversarial)
        print('And in log-scale: %.6f%%' % np.math.log(linear_mmd_real_vs_adversarial))

    # Average over number of iterations
    overall_attack_success /= 5
    avg_l1_perturbation /= 5
    avg_l2_perturbation /= 5
    avg_lInf_perturbation /= 5

    print("\nAverage L1-norm perturbation from L-inf attack for dropout %.2f%%: %.4f%%"
          % (dropout, avg_l1_perturbation))
    print("\nAverage L2-norm perturbation from L-inf attack for dropout %.2f%%: %.4f%%"
          % (dropout, avg_l2_perturbation))
    print("\nAverage LInf-norm perturbation from L-inf attack for dropout %.2f%%: %.4f%%"
          % (dropout, avg_lInf_perturbation))
    print("Overall success of the L-infinity attack on dropout level %.2f%% with LInf bound %.3f: %.3f%%" %
          (dropout, eps, overall_attack_success))
