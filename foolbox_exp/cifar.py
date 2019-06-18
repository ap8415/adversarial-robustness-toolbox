import sys
from os.path import abspath

sys.path.append(abspath('.'))

from experiment_models import neural_networks, convolutional
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

from art.utils import load_cifar10
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

np.set_printoptions(threshold=sys.maxsize)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

(x_train, y_train), (x_test, y_test), min_, max_ = load_cifar10()

import time

print("Current time: %.2f" % time.time())

dr = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
# dr = [0.4]

l1_std = []
l2_std = []
linf_std = []

l1_normalized = []
l2_normalized = []
linf_normalized = []

l1_misclas = []
l2_misclas = []
linf_misclas = []

failure_rate = []

linear_mmd = []

acc = []

for dropout in dr:
    kmodel = None

    kmodel = convolutional.mini_VGG_foolbox(dropout, dropout, 0, "cifar10")

    # kmodel.fit(x_train, y_train, epochs=1, batch_size=128)
    kmodel.fit(x_train, y_train, epochs=100, batch_size=128)

    preds = np.argmax(kmodel.predict(x_test), axis=1)

    attack = CarliniWagnerL2Attack(kmodel, Misclassification())

    # x_sample = x_test[:10]
    # y_sample = y_test[:10]
    x_sample = x_test[:1000]
    y_sample = y_test[:1000]

    adversarial = attack(x_sample, np.argmax(y_sample, axis=1), binary_search_steps=5, max_iterations=600)

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
                adv_examples_no_misclassification.append(x_sample[i])
                orig_examples_no_misclassification.append(adversarial[i])
        else:
            failed += 1
            orig_examples_failed.append(x_sample[i])
            correct_labels_failed.append(y_sample[i])

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
                adv_examples.append(adversarial_strong[i])
                orig_examples.append(orig_examples_failed[i])
                correct_labels.append(correct_labels_failed[i])

                adv_examples_no_misclassification.append(x_sample[i])
                orig_examples_no_misclassification.append(adversarial_strong[i])

                failed -= 1

    print('Failed after the 2nd attack attempt: %d' % failed)
    failure_rate.append(failed)

    adv_examples = np.array(adv_examples)
    correct_labels = np.array(correct_labels)
    orig_examples = np.array(orig_examples)
    perturbations = np.array(perturbations)

    orig_examples_no_misclassification = np.array(orig_examples_no_misclassification)
    adv_examples_no_misclassification = np.array(adv_examples_no_misclassification)

    l1_perturbations = [LA.norm(perturbation.flatten(), 1) for perturbation in perturbations]
    l2_perturbations = [LA.norm(perturbation.flatten(), 2) for perturbation in perturbations]
    lInf_perturbations = [LA.norm(perturbation.flatten(), np.inf) for perturbation in perturbations]
    linear_mmd_real_vs_adversarial = mmd_evaluation(orig_examples_no_misclassification,
                                                    adv_examples_no_misclassification)

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

    # We discount the misclassified samples in this computation.
    misclassified_coef = 1 / ((len(x_sample) - misclassified) / len(x_sample))
    l1_misclas.append(np.average(l1_perturbations) * attack_success_coef * misclassified_coef)
    l2_misclas.append(np.average(l2_perturbations) * attack_success_coef * misclassified_coef)
    linf_misclas.append(np.average(lInf_perturbations) * attack_success_coef * misclassified_coef)

    linear_mmd.append(linear_mmd_real_vs_adversarial)

    print("Current time: %.2f" % time.time())

    acc.append(100 * (len(x_sample) - misclassified) / len(x_sample))
    # count the number of correctly classified samples on the adversarial set

print('Average l1, l2, linf perturbations over the attacks:')
print(l1_std)
print(l2_std)
print(linf_std)

print('Average l1, l2, linf normalized perturbations wrt failed attacks over the successful attacks:')
print(l1_normalized)
print(l2_normalized)
print(linf_normalized)

# We discount the misclassified samples in this computation.
print('Average l1, l2, linf normalized perturbations wrt failed and misclassified over the successful attacks:')
print(l1_misclas)
print(l2_misclas)
print(linf_misclas)

print('No of failures in attack after 2 attempts:')
print(failure_rate)

print('MMD computation for real vs adversarial samples:')
print(linear_mmd)
print('And in log-scale:')
print([np.math.log(x) for x in linear_mmd])

print('Accuracies on the adversarial test set:')
print(acc)