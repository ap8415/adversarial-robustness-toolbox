"""Trains a DNN on the MNIST dataset, then attacks it with the Carlini-Wagner attack."""
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
from os.path import abspath

sys.path.append(abspath('.'))

# dropout 0

avg_fail_eps_0_3 = [9.76, 11.32, 12.1, 11.62, 11.72, 12.6, 12.68,
                    13.72, 12.62, 13.02, 12.78, 13.34, 12.18, 12.5, 12.2, 11.54]
avg_fail_eps_0_275 = [12.08, 12.88, 12.66, 13.6, 15.34, 14.16, 14.94,
                      14.9, 15.9, 15.36, 15.54, 14.96, 14.44, 14.6, 14.36, 13.0]
avg_fail_eps_0_25 = [14.06, 14.00, 14.3, 16.24, 16.4, 15.14, 16.78,
                     15.7, 15.44, 15.9, 14.76, 15.88, 15.2, 15.42, 14.24, 14.48]
avg_fail_eps_0_225 = [14.76, 16.44, 16.38, 18.14, 16.42, 16.98, 17.68,
                      17.24, 16.84, 16.6, 16.6, 15.36, 16.32, 16.26, 15.6, 15.88]
avg_fail_eps_0_2 = [17.38, 17.62, 18.0, 17.66, 18.48, 18.84, 18.8,
                    18.92, 18.9, 18.28, 17.260, 17.9, 18.32, 18.38, 17.04, 17.2]
avg_fail_eps_0_175 = [22.0, 22.34, 21.72, 20.66, 20.82, 21.12, 22.48,
                      20.86, 20.84, 20.46, 20.62, 20.18, 19.98, 19.9, 19.28, 18.68]
avg_fail_eps_0_15 = [22.86, 21.72, 22.94, 22.82, 22.48, 23.36, 22.34,
                     21.88, 22.66, 22.4, 21.7, 20.72, 20.34, 20.44, 19.56, 18.84]
avg_fail_eps_0_125 = [26.26, 25.5, 25.68, 25.64, 27.28, 27.48, 26.36,
                      28.02, 26.4, 26.46, 26.42, 26.32, 26.26, 26.72, 26.42, 24.48]
avg_fail_eps_0_1 = [35.94, 35.3, 35.32, 36.34, 37.28, 35.98, 36.28,
                    38.1, 36.44, 36.6, 36.64, 37.9, 37.82, 37.18, 36.04, 37.82]
avg_fail_eps_0_09 = [41.78, 42.0, 41.0, 41.48, 41.48, 44.28, 43.76,
                     43.7, 44.72, 44.3, 44.16, 44.24, 44.26, 44.54, 43.76, 43.62]
avg_fail_eps_0_08 = [48, 47.16, 47.82, 48.56, 48.46, 49.48, 51.04,
                     51.02, 50.66, 52.74, 50.32, 50.7, 50.76, 51.48, 50.76, 49.72]
avg_fail_eps_0_07 = [55.74, 56, 55.52, 59.18, 59.22, 60.52, 60.0,
                     61.18, 60.58, 60.38, 61.8, 62.82, 61.2, 61.44, 61.4, 60.72]
avg_fail_eps_0_06 = [66.36, 66.5, 66.26, 68.36, 70.8, 70.9, 71.44,
                     71.48, 72.28, 72.18, 71.7, 73.46, 73.9, 72.4, 72.04, 73.1]
avg_fail_eps_0_05 = [75.84, 76.26, 76.62, 78.3, 79.24, 79.04, 80.12,
                     80.0, 79.39, 80.94, 81.22, 80.98, 82.14, 80.3, 80.5, 80.48]
