import sys
from os.path import abspath

sys.path.append(abspath('.'))

import matplotlib.pyplot as plt


fig = plt.figure()

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


plt.plot(dropout_levels, l2_min,
         dropout_levels, l2_mean,
         dropout_levels, l2_max)
fig.suptitle('6-layer DNN, Carlini attack on MNIST')
plt.xlabel('Dropout%')
plt.ylabel('Security score')
plt.savefig('figura.png')

# plt.plot(dropout_levels, linf_min,
#          dropout_levels, linf_mean,
#          dropout_levels, linf_max)
# fig.suptitle('3-layer DNN, untargeted PGD attack on MNIST')
# plt.xlabel('Dropout%')
# plt.ylabel('Security score')
# plt.savefig('figura.png')



