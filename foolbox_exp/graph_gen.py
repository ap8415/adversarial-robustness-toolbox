import sys
from os.path import abspath

sys.path.append(abspath('.'))

import argparse
import numpy as np
import numpy.linalg as LA
import tensorflow as tf
import matplotlib.pyplot as plt

from art.attacks.carlini import CarliniL2Method
from art.utils import load_mnist_vectorized, load_mnist
from experiment_models import neural_networks, convolutional
from experiment_models.utils import mmd_evaluation
from statistics import mean
from keras.backend.tensorflow_backend import set_session


fig = plt.figure()

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

l1_metric = [14.7519922273622, 15.359281555143685, 15.627670277936458, 15.27971358473565,
             15.730460663369158, 15.046738201430395, 15.504061415554652, 15.830936674689335,
             15.432925970605964, 15.77317169037623, 14.521927570395537, 14.829866380877016,
             14.51221157978095, 14.135752031862191, 13.67340174816327, 12.508636105853565,
             10.166936475091276, 9.770728835828297]

[14.974243668009406, 15.443470465688545, 15.64489741893764, 15.738540475048746,
 15.457201378584386, 15.610316463310454, 15.175122449347008, 15.13097540047086,
 15.124844783900482, 15.981599479161197, 15.140595346189905, 14.482685555726656,
 14.661745078507499, 14.12278647477502, 13.483347551176795, 13.237602547052948,
 11.740370141652242, 12.842273601234227]

l1_min = [14.7519922273622, 15.259281555143685, 15.627670277936458, 15.27971358473565,
          15.457201378584386]


l2_metric = [1.078364058293457, 1.1513407141391439, 1.148011047238633, 1.1186917128472362,
             1.1399812312428312, 1.0876664689889906, 1.1279361430571613, 1.1647648411234748,
             1.1125496174174863, 1.098730778732488, 1.0379450698512702, 1.0603500516627278,
             1.0135491389306603, 0.9581720596661039, 0.9141356712790083, 0.8396848583025135,
             0.6580497696112486, 0.6142328387171366]

linf_metric = [0.2432650282567314, 0.27650753584423765, 0.2644987712525123, 0.2576821342862329,
               0.2758840562627954, 0.2614846043733805, 0.2701722551101496, 0.2964142214373986,
               0.2608121615031022, 0.2616200446061689, 0.26488936252959866, 0.26385502746145567,
               0.23566267916215486, 0.21341420050321816, 0.20040100766669702, 0.16771407182773607,
               0.1262471579798847, 0.10647661789061116]




plt.plot(dropout_levels, min_l1_perturbation,
         dropout_levels, mean_l1_perturbation,
         dropout_levels, max_l1_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, L1 distance')
plt.xlabel('Dropout%')
plt.ylabel('L1 distance')
plt.savefig(args.experiment_type + '_l1_x.png')

fig = plt.figure()
plt.plot(dropout_levels, min_l2_perturbation,
         dropout_levels, mean_l2_perturbation,
         dropout_levels, max_l2_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, L2 distance')
plt.xlabel('Dropout%')
plt.ylabel('L2 distance')
plt.savefig(args.experiment_type + '_l2_x.png')

fig = plt.figure()
plt.plot(dropout_levels, min_lInf_perturbation,
         dropout_levels, mean_lInf_perturbation,
         dropout_levels, max_lInf_perturbation)
fig.suptitle(actual_names[args.experiment_type] + ' on MNIST, LInf distance')
plt.xlabel('Dropout%')
plt.ylabel('LInf distance')
plt.savefig(args.experiment_type + '_lInf_x.png')