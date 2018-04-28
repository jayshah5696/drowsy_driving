# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 21:08:12 2018

@author: Lenovo
"""

#loading dataset in spydet python IDE
import pandas as pd

train =pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

train.head()
train=train.drop(train.columns[[0]], axis=1)
test=test.drop(test.columns[[0]],axis=1)

ytest=test.iloc[:,1]
ytrain=train.iloc[:,1]

xtrain=train.iloc[:,3:1803]
xtest=test.iloc[:,3:1803]


xtrain=xtrain.values
xtest=xtest.values


x_train =xtrain.reshape(xtrain.shape[0],600,3)
x_test=xtest.reshape(xtest.shape[0],600,3)

#standardising data
import tensorflow as tf
scaler=tf.nn.l2_normalise(x_train,dim=1,epsilon=1e-12,name=None)

#importing tensorflow and keras
# and making sure they run code on gpu
import keras
from keras import backend as K

num_cores=4
GPU:
    num_GPU=1
    num_CPU=1
if CPU:
    num_CPU=1
    num_GPU=0

config= tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                        device_count={'CPU':1,'GPU':1})

session=tf.Session(config=config)
K.set_session(session)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())