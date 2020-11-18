# -*- coding: utf-8 -*-
'''切断冪基底を図示'''

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


x = np.linspace(0, 10, 100)
for i in range(4):
    y = x ** i
    print(type(y))
    exit
    plt.plot(x, y)
    plt.savefig('h' + str(i) + '.png')
    plt.clf()

for i in range(1, 5):
    ci = 10 * i / 5
    y =  (x-ci)**3
    plt.plot(x, y)
    plt.savefig('h' + str(i+3) + '.png')
    plt.clf()
