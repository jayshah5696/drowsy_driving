# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 23:18:25 2018

@author: Lenovo
"""

from pyts.transformation import GASF, GADF
from pyts import transformation, classification, visualization
gasf = GASF(image_size=24, overlapping=False, scale='-1')
X_gasf = gasf.transform(X_standardized)

gadf = GADF(image_size=24, overlapping=False, scale='-1')
X_gadf = gadf.transform(X_standardized)
from pyts.visualization import plot_standardscaler

plot_standardscaler(X[0])


from pyts.visualization import plot_gasf
from pyts.visualization import plot_gadf
plot_gasf(X_standardized[0], image_size=48, overlapping=False, scale='-1')
plot_gadf(X_standardized[0], image_size=96, overlapping=False, scale='-1')
from __future__ import division




import numpy as np
from scipy.stats import norm
	
n_samples = 10
n_features = 48
n_classes = 2
	
delta = 0.5
dt = 1
x = 0.
	
X = np.zeros((n_samples, n_features))
X[:, 0] = x
	
for i in range(n_samples):
    start = x
    for k in range(1, n_features):
        start += norm.rvs(scale=delta**2 * dt)
        X[i][k] = start
	
y = np.random.randint(n_classes, size=n_samples)
from pyts.transformation import StandardScaler

standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(X)



from pyts.transformation import MTF

mtf = MTF(image_size=61, n_bins=1, quantiles='empirical', overlapping=False)
X_mtf = mtf.transform(X_standardized)

from pyts.visualization import plot_mtf
plot_mtf(X_standardized[0], image_size=61, n_bins=4, quantiles='empirical', overlapping=False)







plot_mtf(X_standardized[0], image_size=61, n_bins=10, quantiles='empirical', overlapping=False)

plot_gasf(X_standardized[0], image_size=61, overlapping=False, scale='-1')

plot_gadf(X_standardized[0], image_size=30, overlapping=False, scale='-1')