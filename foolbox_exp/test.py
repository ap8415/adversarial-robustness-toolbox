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

dr = [0]#, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

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

    kmodel.fit(x_train, y_train, epochs=10, batch_size=128)
    # kmodel.fit(x_train, y_train, epochs=50, batch_size=128)

    preds = np.argmax(kmodel.predict(x_test), axis=1)
    x_sample = np.take(x_test, ones, axis=0)[:5]
    y_sample = np.array([1 for x in x_sample])
    y_target = np.array([to_one_hot(7) for _ in x_sample])

    attack = CarliniWagnerL2Attack(kmodel, TargetClass(7))
    # attack = RandomPGD(kmodel, TargetClass(7))

    adversarial = attack(x_sample, y_sample, binary_search_steps=5, max_iterations=600)
    # adversarial = attack(x_sample, y_sample, iterations=30)

    print(kmodel.predict(adversarial))

    # For those samples for which the L2 method does not produce an adversarial sample within the attack parameters,
    # we exclude them from the perturbation evaluation.

