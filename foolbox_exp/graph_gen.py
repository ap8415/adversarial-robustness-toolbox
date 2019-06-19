import sys
from os.path import abspath

sys.path.append(abspath('.'))

import matplotlib.pyplot as plt


fig = plt.figure()

# dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

# dropout_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


dropout_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85]


l2_min = [1.0632, 1.4355, 1.5381, 1.6731, 1.9437,
          1.9429, 2.0548, 2.3653, 2.6113, 2.3452,
          2.2115]

l2_mean = [1.0844, 1.6021, 1.6432, 1.7576, 2.0701,
           2.0984, 2.2444, 2.5395, 2.9814, 2.7143,
           2.5134]

l2_max = [1.0944, 1.7706, 1.7945, 1.8751, 2.1620,
          2.1918, 2.4165, 2.8149, 3.3471, 3.1140,
          3.1035]

#
plt.plot(dropout_levels, l2_min,
         dropout_levels, l2_mean,
         dropout_levels, l2_max)
fig.suptitle('LeNet5, targeted Carlini attack on MNIST')
plt.xlabel('Dropout%')
plt.ylabel('Security score')
plt.savefig('figura.png')



# plt.plot(dropout_levels, linf_min,
#          dropout_levels, linf_mean,
#          dropout_levels, linf_max)
# fig.suptitle('LeNet5, untargeted PGD attack on MNIST')
# plt.xlabel('Dropout%')
# plt.ylabel('Security score')
# plt.savefig('vgg_pooling.png')



