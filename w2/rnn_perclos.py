# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 17:49:09 2018

@author: Lenovo
"""
#perclos RNN

import pandas as pd

train =pd.read_csv('train_balanced_scaled.csv')
test=pd.read_csv('test_trimmed_scaled.csv')

train.head()
train=train.drop(train.columns[[0]], axis=1)
test=test.drop(test.columns[[0]],axis=1)

ytest=test.iloc[:,0]
ytrain=train.iloc[:,0]

y_train=ytrain.values
y_test=ytest.values
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test=to_categorical(y_test)


xtrain=train.iloc[:,1:62]
xtest=test.iloc[:,1:184]

xtrain=xtrain.values
xtest=xtest.values


x_train =xtrain.reshape(xtrain.shape[0],61,1)
x_test=xtest.reshape(xtest.shape[0],61,1)

#loading Required Libraries
import tensorflow as tf
import keras as ks
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM


model=Sequential()
model.add(LSTM(61,input_shape=(61, 1)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)


score = model.evaluate(x_test, y_test, verbose=0)
y=model.predict_proba(x_test)
from sklearn import  metrics
j=metrics.roc_auc_score(y_test, y)
j=np.array(j)
j
#0.653



model1=Sequential()
model1.add(LSTM(61,input_shape=(61, 1),kernel_initializer='normal'))
model1.add(Dropout(0.2))
model1.add(Dense(2, activation='softmax'))

model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.summary()

model1.fit(x_train, y_train, epochs=2, batch_size=1, verbose=1)

score = model1.evaluate(x_test, y_test, verbose=0)
y=model1.predict_proba(x_test)

j=metrics.roc_auc_score(y_test, y)
j=np.array(j)
j
#0.59


model2=Sequential()
model2.add(LSTM(61,input_shape=(61, 1),kernel_initializer='normal'))
model2.add(Dropout(0.2))
model2.add(Activation('relu'))
model2.add(Dense(2, activation='softmax'))

model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

model2.fit(x_train, y_train,validation_data=(x_test,y_test) ,epochs=2 ,batch_size=16, verbose=1)
score = model2.evaluate(x_test, y_test, verbose=0)
y=model2.predict_proba(x_test)

j=metrics.roc_auc_score(y_test, y)
j=np.array(j)


model3=Sequential()
model3.add(LSTM(61,input_shape=(61, 1),kernel_initializer='normal'))
model3.add(Dropout(0.4))
model3.add(Activation('relu'))
model3.add(Dense(61, activation='relu'))
model3.add(Dropout(0.4))
model3.add(Dense(2, activation='softmax'))

model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['cosine_proximity','accuracy'])
model3.summary()

model3.fit(x_train, y_train,validation_data=(x_test,y_test) ,epochs=5 ,batch_size=8, verbose=1)
score = model2.evaluate(x_test, y_test, verbose=0)
y=model3.predict_proba(x_test)

j=metrics.roc_auc_score(y_test, y)
j=np.array(j)
j