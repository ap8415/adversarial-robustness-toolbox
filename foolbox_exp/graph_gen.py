import sys
from os.path import abspath

sys.path.append(abspath('.'))

import matplotlib.pyplot as plt


fig = plt.figure()

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]


linf_min = [0.0961, 0.1031, 0.0985, 0.1100, 0.1125,
            0.1212, 0.1624, 0.1834, 0.1800, 0.1912,
            0.1931, 0.1899, 0.1736, 0.1663, 0.1676,
            0.1959, 0.2476, 0.2231]

linf_mean = [0.0978, 0.1122, 0.1094, 0.1170, 0.1179,
             0.1315, 0.1755, 0.2019, 0.2151, 0.2207,
             0.2212, 0.2094, 0.1922, 0.1821, 0.1763,
             0.2152, 0.2799, 0.2361]

linf_max = [0.1004, 0.1216, 0.1198, 0.1261, 0.1255,
            0.1473, 0.1938, 0.2343, 0.2461, 0.2473,
            0.2433, 0.2312, 0.2144, 0.2156, 0.1908,
            0.2416, 0.3300, 0.2519]



#
# plt.plot(dropout_levels, l2_min,
#          dropout_levels, l2_mean,
#          dropout_levels, l2_max)
# fig.suptitle('LeNet5, Carlini attack on MNIST')
# plt.xlabel('Dropout%')
# plt.ylabel('Security score')
# plt.savefig('figura.png')

plt.plot(dropout_levels, linf_min,
         dropout_levels, linf_mean,
         dropout_levels, linf_max)
fig.suptitle('6-layer DNN, untargeted PGD attack on MNIST')
plt.xlabel('Dropout%')
plt.ylabel('Security score')
plt.savefig('figura.png')



