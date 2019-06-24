import sys
from os.path import abspath
from statistics import mean

sys.path.append(abspath('.'))

from models import deep, convolutional
from models.metrics import to_one_hot

from foolbox.models import KerasModel
from foolbox.batch_attacks import CarliniWagnerL2Attack, RandomPGD
from foolbox.criteria import TargetClass

import argparse
import numpy as np
import numpy.linalg as LA
import logging
import matplotlib.pyplot as plt
import tensorflow as tf

from models.utils import load_mnist, load_mnist_vectorized, load_cifar10
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Experiment parameters.')
parser.add_argument("experiment_type", help="The model type used by the experiment. "
                                            "Values: three_layer, five_layer, six_layer, leNet, VGG")
parser.add_argument("attack_type", help="The norm attacked. Values: l2 or linf.")
parser.add_argument("dataset", help="The dataset used by the experiment. Values: mnist, cifar.")
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

if args.dataset == 'mnist':
    if args.experiment_type in ["three_layer", "five_layer", "six_layer"]:
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist_vectorized()
    elif args.experiment_type in ["VGG", "leNet"]:
        (x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()
        # Pad images to 32x32 size
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
elif args.dataset == 'cifar':
    (x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()
else:
    raise Exception("Invalid dataset!")

input_shape = x_train.shape[1:]
output_shape = len(y_train[0])

ones = np.where(np.argmax(y_test, axis=1) == 1)[0]

dropout_fractions = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

l1_std = []
l2_std = []
linf_std = []

l1_normalized = []
l2_normalized = []
linf_normalized = []

l1_std_min = []
l2_std_min = []
linf_std_min = []

l1_normalized_min = []
l2_normalized_min = []
linf_normalized_min = []

l1_std_max = []
l2_std_max = []
linf_std_max = []

l1_normalized_max = []
l2_normalized_max = []
linf_normalized_max = []

failure_rate = []

for dropout in dropout_fractions:
    l1_all = []
    l2_all = []
    linf_all = []

    l1_all_normalized = []
    l2_all_normalized = []
    linf_all_normalized = []

    failure_all = []
    for i in range(5):
        model = None

        if args.experiment_type == "three_layer":
            model = deep.three_layer_dnn(input_shape, output_shape, dropout, 0, 0)
        elif args.experiment_type == "five_layer":
            model = deep.five_layer_dnn_model_wide(input_shape, output_shape, dropout, 0, 0)
        elif args.experiment_type == "six_layer":
            model = deep.six_layer_dnn_model_wide(input_shape, output_shape, dropout, 0, 0)
        elif args.experiment_type == "VGG":
            model = convolutional.vgg_model_wide(args.dataset, dropout, 0, 0)
        elif args.experiment_type == "leNet":
            model = convolutional.leNet_model_wide(dropout, 0, 0)
        else:
            raise Exception("Invalid model!")

        model.fit(x_train, y_train, epochs=50, batch_size=128)
        preds = np.argmax(model.predict(x_test), axis=1)

        kmodel = KerasModel(model=model, bounds=(min_, max_))

        attack = None
        if args.attack_type == 'l2':
            attack = CarliniWagnerL2Attack(kmodel, TargetClass(7))
        elif args.attack_type == 'linf':
            attack = RandomPGD(kmodel, TargetClass(7))

        x_sample = np.take(x_test, ones, axis=0)

        # We exclude by default those examples which are not predicted by the classifier as 1s.
        true_ones = np.where(preds == 1)[0]

        x_sample = np.take(x_sample, true_ones, axis=0)
        y_sample = np.array([to_one_hot(1) for _ in x_sample])

        adversarial = None
        if args.attack_type == 'l2':
            adversarial = attack(x_sample, np.argmax(y_sample, axis=1), binary_search_steps=5, max_iterations=600)
        else:
            adversarial = attack(x_sample, np.argmax(y_sample, axis=1), iterations=30)

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
        for j in range(0, len(adversarial)):
            # If the attack fails, an array of NaN's is returned. We check for this here.
            if not any([np.isnan(x) for x in adversarial[j].flatten()]):
                perturbations.append(np.abs(adversarial[j] - x_sample[j]))
                adv_examples.append(adversarial[j])
                orig_examples.append(x_sample[j])
                correct_labels.append(y_sample[j])

                # We count the misclassified samples as well, as those are returned as adversarials equal to the input
                # image. All misclassified samples will reach this if-branch by definition of the adversarial attack.
                if np.argmax(y_sample[j]) != preds[j]:
                    misclassified += 1
                else:
                    adv_examples_no_misclassification.append(x_sample[j])
                    orig_examples_no_misclassification.append(adversarial[j])
            else:
                failed += 1
                orig_examples_failed.append(x_sample[j])
                correct_labels_failed.append(y_sample[j])

        print('Initially failed: %d' % failed)
        print('Misclassified: (should be 0!):    %d' % misclassified)

        orig_examples_failed = np.array(orig_examples_failed)
        correct_labels_failed = np.array(correct_labels_failed)

        # If the attack failed on any samples, we retry the attack with increased bounds on binary search and
        # number of iterations to ensure success.
        # If on a sample the attack fails even with these parameters, the sample is extremely resilient to attacks, and
        # we stop trying to attack it and instead incorporate it into our metrics.
        if len(orig_examples_failed) > 0:
            if args.attack_type == 'l2':
                adversarial_strong = attack(orig_examples_failed, np.argmax(correct_labels_failed, axis=1),
                                            binary_search_steps=15, max_iterations=1000)
            else:
                adversarial_strong = attack(orig_examples_failed, np.argmax(correct_labels_failed, axis=1),
                                            iterations=75)

            for j in range(0, len(adversarial_strong)):
                # If the attack fails, an array of NaN's is returned.
                # Since it failed on the second try, we abandon trying to compute an adversarial sample for this run.
                if not any([np.isnan(x) for x in adversarial_strong[j].flatten()]):
                    perturbations.append(np.abs(adversarial_strong[j] - orig_examples_failed[j]))
                    adv_examples.append(adversarial_strong[j])
                    orig_examples.append(orig_examples_failed[j])
                    correct_labels.append(correct_labels_failed[j])

                    adv_examples_no_misclassification.append(adversarial_strong[j])
                    orig_examples_no_misclassification.append(orig_examples_failed[j])

                    failed -= 1

        print('Failed after the 2nd attack attempt: %d' % failed)

        adv_examples = np.array(adv_examples)
        correct_labels = np.array(correct_labels)
        orig_examples = np.array(orig_examples)
        perturbations = np.array(perturbations)

        l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
        l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
        lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]

        l1_all.append(np.average(l1_perturbations))
        l2_all.append(np.average(l2_perturbations))
        linf_all.append(np.average(lInf_perturbations))

        # We normalize by the coefficient of failed attacks to incorporate failure rate into our security measure.
        # For example, if 50% of our attacks fail, we multiply by a coeficient of (1/ (50/100)) = 1/ (1/2) = 2.
        # success_coef = ((10 - failed) / 10)
        attack_success_coef = 1 / ((len(x_sample) - failed) / len(x_sample))

        l1_all_normalized.append(np.average(l1_perturbations) * attack_success_coef)
        l2_all_normalized.append(np.average(l2_perturbations) * attack_success_coef)
        linf_all_normalized.append(np.average(lInf_perturbations) * attack_success_coef)

        failure_all.append(failed / len(x_sample))

    l1_std.append(mean(l1_all))
    l2_std.append(mean(l2_all))
    linf_std.append(mean(linf_all))

    l1_normalized.append(mean(l1_all_normalized))
    l2_normalized.append(mean(l2_all_normalized))
    linf_normalized.append(mean(linf_all_normalized))

    l1_std_min.append(min(l1_all))
    l2_std_min.append(min(l2_all))
    linf_std_min.append(min(linf_all))

    l1_normalized_min.append(min(l1_all_normalized))
    l2_normalized_min.append(min(l2_all_normalized))
    linf_normalized_min.append(min(linf_all_normalized))

    l1_std_max.append(max(l1_all))
    l2_std_max.append(max(l2_all))
    linf_std_max.append(max(linf_all))

    l1_normalized_max.append(max(l1_all_normalized))
    l2_normalized_max.append(max(l2_all_normalized))
    linf_normalized_max.append(max(linf_all_normalized))

    failure_rate.append(mean(failure_all))

print('Printing statistics...')

print('Average l1, l2, linf perturbations over the attacks:')
print(l1_std)
print(l2_std)
print(linf_std)

print('Average l1, l2, linf normalized perturbations wrt failed attacks over the successful attacks:')
print(l1_normalized)
print(l2_normalized)
print(linf_normalized)

print('Fraction of failures in attacks:')
print(failure_rate)

fig = plt.figure()

name_map = {'three_layer': '3-layer DNN', 'five_layer': '5-layer DNN', 'six_layer': '6-layer DNN',
            'leNet': 'LeNet5', 'VGG': 'VGG'}
attack_map = {'l2': 'Carlini', 'linf': 'PGD'}
dataset_map = {'mnist': 'MNIST', 'cifar': 'CIFAR10'}

if args.attack_type == 'l2':
    plt.plot(dropout_fractions, l2_normalized_min)
else:
    plt.plot(dropout_fractions, linf_normalized_min)

if args.attack_type == 'l2':
    plt.plot(dropout_fractions, l2_normalized)
else:
    plt.plot(dropout_fractions, linf_normalized)

if args.attack_type == 'l2':
    plt.plot(dropout_fractions, l2_normalized_max)
else:
    plt.plot(dropout_fractions, linf_normalized_max)

figname = '%s, targeted %s attack on %s' % \
    (name_map[args.experiment_type], attack_map[args.attack_type], dataset_map[args.dataset])
fig.suptitle(figname)
plt.xlabel('Dropout%')
plt.ylabel('Security score')
plt.savefig('%s_%s_%s_targeted.png' % (args.experiment_type, args.attack_type, args.dataset))
