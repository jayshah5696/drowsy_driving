# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 15:10:53 2018

@author: Lenovo
"""

import numpy as np
import pandas as pd
import pyts

train=pd.read_csv('train_visual.csv')
#train=train.drop(train.columns[[0]], axis=1)

xtrain=train.iloc[:,2:63]
xtrain=xtrain.values


from pyts.transformation import StandardScaler

#standardscaler = StandardScaler(epsilon=1e-2)
#X_standardized = standardscaler.transform(xtrain)
X_standardized=pd.read_csv('X_standardized.csv')
X_standardized=X_standardized.iloc[:,1:63]
X_standardized=X_standardized.values


from pyts.transformation import GASF, GADF
from pyts import transformation, classification, visualization


from pyts.visualization import plot_gasf

print("Grammian Angular Summation Field-- Images")
print("-----------------------------------------------------------")
for i in range (0,80):
    plot_gasf(X_standardized[i], image_size=61, overlapping=False, scale='-1')
    print(i,train['RunName'][i],"Drowsy--",train['drowsy'][i],"Window Number--",train['win60s'][i])
    print("----------------------------")



from pyts.visualization import plot_gadf

print("Grammian Angular Differentiate Field-- Images")
print("-----------------------------------------------------------")
for i in range (0,80):
    plot_gadf(X_standardized[i], image_size=61, overlapping=False, scale='-1')
    print(i,train['RunName'][i],"Drowsy--",train['drowsy'][i],"Window Number--",train['win60s'][i])
    print("----------------------------")


from pyts.visualization import plot_mtf


print("Markovian Transition Field-- Images")
print("-----------------------------------------------------------")
for i in range (0,80):
    plot_mtf(X_standardized[i], image_size=61, n_bins=4, quantiles='empirical', overlapping=False)
    print(i,train['RunName'][i],"Drowsy--",train['drowsy'][i],"Window Number--",train['win60s'][i])
    print("----------------------------")

