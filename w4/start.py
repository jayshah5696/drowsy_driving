# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:41:26 2018

@author: Lenovo
"""

import os
os.chdir('E:\\USA\\Projects\\Research\\R_code\\w2')
#################################################################################################
#loading Libraries
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten	
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils import to_categorical
from pyts.transformation import StandardScaler
from pyts.transformation import GASF, GADF, MTF
from pyts import transformation, classification, visualization
from pyts.visualization import plot_gasf
from pyts.visualization import plot_gadf
from sklearn import  metrics
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
#from pandas_ml import ConfusionMatrix
from tensorflow import set_random_seed
from keras.models import load_model
#seeding 1001
seed = 1001
np.random.seed(seed) #for numpy
set_random_seed(seed) #for Tensorflow   
################################################################################################



train =pd.read_csv('train_balanced.csv')
test=pd.read_csv('test_final.csv')
train=train.drop(train.columns[[0]], axis=1)
test=test.drop(test.columns[[0]],axis=1)

###########################################################################################

#loading Training Data
xtrain=train.iloc[:,1:62]
xtrain=np.array(xtrain.values)
ytrain=train.iloc[:,0]
ytrain=np.array(ytrain.values)

#loading Test data
xtest=test.iloc[:,3:64]
xtest=np.array(xtest.values)
ytest=test.iloc[:,2]
ytest=np.array(ytest.values)


y_train = to_categorical(ytrain)
y_test=to_categorical(ytest)

#reshaping time series window
standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(xtrain)
Xt_standardized = standardscaler.transform(xtest)





#defining custome metric for guiding our neural network
def auc_roc(y_true, y_pred):
    # any tensorflow metric
    #value, update_op = metrics.roc_auc_score(y_true, y_pred)
    value, update_op = tf.contrib.metrics.streaming_auc(y_pred, y_true)

    # find all variables created for this metric
    metric_vars = [i for i in tf.local_variables() if 'auc_roc' in i.name.split('/')[1]]

    # Add metric variables to GLOBAL_VARIABLES collection.
    # They will be initialized for new session.
    for v in metric_vars:
        tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, v)

    # force to update metric values
    with tf.control_dependencies([update_op]):
        value = tf.identity(value)
        return value



#metrics.roc_auc_score(y_true, y_pred)





##########################################################################################
#building model for mtf transformation
mtf = MTF(image_size=61, n_bins=6, quantiles='empirical', overlapping=False)
X_mtf = mtf.transform(X_standardized)
Xt_mtf=mtf.transform(Xt_standardized)

x_train =X_mtf.reshape(X_mtf.shape[0],61,61,1)
x_test =Xt_mtf.reshape(Xt_mtf.shape[0],61,61,1)
