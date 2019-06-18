import math
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
print("\n\n\n%s\n\n\n", args.experiment_type)

l1_regularization = [1e-6, 2e-6, 4e-6, 7e-6, 1e-5, 2e-5, 3e-5, 5e-5, 7.5e-5, 1e-4, 1.25e-4, 1.5e-4, 2e-4, 3e-4, 5e-4]

for l1_reg in l1_regularization:
    kmodel = None

    print("Regularization lambda: %.8f" % l1_reg)

    if args.experiment_type == "three_layer_dnn":
        kmodel = neural_networks.three_layer_dnn_foolbox(x_train.shape[1:], 300, 100, 0, l1_reg, 0)
    elif args.experiment_type == "five_layer_dnn":
        kmodel = neural_networks.symmetric_five_layer_nn_l1reg_foolbox(x_train.shape[1:], l1_reg, l1_reg)
    elif args.experiment_type == "six_layer_dnn":
        kmodel = neural_networks.asymmetric_six_layer_nn_l1reg_foolbox(x_train.shape[1:], l1_reg)
    elif args.experiment_type == "VGG":
        kmodel = convolutional.mini_VGG_foolbox(0, 0, l1_reg, "mnist")
    elif args.experiment_type == "leNet5":
        kmodel = convolutional.leNet_cnn_l1reg_foolbox(l1_reg, "mnist")

    acc = 0
    for i in range(0, 10):
        kmodel.fit(x_train, y_train, epochs=5, batch_size=128)
        acc = np.sum(np.argmax(kmodel.predict(x_test), axis=1) == np.argmax(y_test, axis=1))
        print(acc)
    print("Final accuracy: %d" % acc)

    total_weights = 0
    small_weights_4 = 0
    small_weights_5 = 0
    small_weights_6 = 0
    small_weights_7 = 0

    for layer in kmodel.model.layers:
        weights = layer.get_weights()[0] # ignore bias vector
        # print(weights)
        total_weights += len(weights.flatten())
        for w in weights.flatten():
            if math.fabs(w) <= 1e-4:
                small_weights_4 += 1
            if math.fabs(w) <= 1e-5:
                small_weights_5 += 1
            if math.fabs(w) <= 1e-6:
                small_weights_6 += 1
            if math.fabs(w) <= 1e-7:
                small_weights_7 += 1

    print("Total weights: %d; smaller than 10^-4: %d'; smaller than 10^-5: %d; smaller than 10^-6: %d; smaller than"
          " 10^-7: %d" % (total_weights, small_weights_4, small_weights_5, small_weights_6, small_weights_7))

    print("Current time: %.2f" % time.time())
