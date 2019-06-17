import math
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

l1_regularization = [5e-4]


for l1_reg in l1_regularization:
    kmodel = None

    if args.experiment_type == "three_layer_dnn":
        kmodel = neural_networks.three_layer_dnn_foolbox(x_train.shape[1:], 300, 100, 0, l1_reg, 0)
    elif args.experiment_type == "five_layer_dnn":
        kmodel = neural_networks.symmetric_five_layer_nn_foolbox(x_train.shape[1:], 0, 0)
    elif args.experiment_type == "six_layer_dnn":
        kmodel = neural_networks.asymmetric_six_layer_nn_foolbox(x_train.shape[1:], 0, 0)
    # elif args.experiment_type == "VGG":
    #     classifier = convolutional.mini_VGG(dropout_levels[dropout], "mnist")
    # elif args.experiment_type == "leNet5":
    #     classifier = convolutional.leNet_cnn_single(dropout_levels[dropout])

    for i in range(0, 10):
        kmodel.fit(x_train, y_train, epochs=4, batch_size=128)
        acc = np.sum(np.argmax(kmodel.predict(x_test), axis=1) == np.argmax(y_test, axis=1))
        print(acc)

    total_weights = 0
    zero_weights = 0

    for layer in kmodel.model.layers:
        weights = layer.get_weights()[0] # ignore bias vector
        # print(weights)
        total_weights += len(weights.flatten())
        for w in weights.flatten():
            if math.fabs(w) < 1e-4:
                zero_weights += 1

    print(total_weights)
    print(zero_weights)


    print("Current time: %.2f" % time.time())
