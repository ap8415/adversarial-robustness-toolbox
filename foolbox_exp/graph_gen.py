import sys
from os.path import abspath

sys.path.append(abspath('.'))

import matplotlib.pyplot as plt


fig = plt.figure()

dropout_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]

[4.602683457277321, 4.629070225249031, 4.682535953013035, 4.829198891242909, 4.57244178648047,
 4.834771115149541, 4.688718115061125, 4.839491594646835, 4.786685898791224, 4.347531566994757,
 4.40929413750907, 4.697692269835482, 4.556300089866203, 4.647025095182523, 4.335377739823021,
 4.435541477925971, 4.575568252963059, 4.429956556447819]

l2_min = [4.473, 4.468, 4.502, 4.535, 4.522,
          4.553, 4.591, 4.619, 4.629, 4.447,
          4.452, 4.512, 4.501, 4.495, 4.335,
          4.345, 4.367, 4.326]

l2_mean = [4.527, 4.569, 4.601, 4.653, 4.678,
           4.693, 4.684, 4.704, 4.719, 4.631,
           4.642, 4.648, 4.621, 4.621, 4.573,
           4.548, 4.54, 4.471]

l2_max = [4.602, 4.639, 4.682, 4.829, 4.784,
          4.834, 4.861, 4.839, 4.833, 4.769,
          4.79, 4.781, 4.767, 4.793, 4.661,
          4.671, 4.702, 4.591]






plt.plot(dropout_levels, l2_min,
         dropout_levels, l2_mean,
         dropout_levels, l2_max)
fig.suptitle('3-layer DNN, Carlini attack on Spambase')
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



