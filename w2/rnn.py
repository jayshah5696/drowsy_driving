# -*- coding: utf-8 -*#loading dataset in spydet python IDE

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


xtrain=train.iloc[:,1:184]
xtest=test.iloc[:,1:184]

xtrain=xtrain.values
xtest=xtest.values


x_train =xtrain.reshape(xtrain.shape[0],61,3)
x_test=xtest.reshape(xtest.shape[0],61,3)

#loading Required Libraries
import tensorflow as tf
import keras as ks
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM


#seeding your neural network
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

model=Sequential()
model.add(LSTM(61,input_shape=(61, 3)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)


score = model.evaluate(x_test, y_test, verbose=0)
y=model.predict_proba(x_test)
from sklearn import  metrics
j=metrics.roc_auc_score(y_test, y)
j=np.array(j)



model=Sequential()
model.add(LSTM(61,input_shape=(61, 3),kernel_initializer='normal'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)




#def auc(y_true,y_pred):
#    j= metrics.roc_auc_score(y_true, y_pred)
#    j=np.array(j)
#    return j


model1=Sequential()
model1.add(LSTM(61,input_shape=(61, 3),return_sequences=False))
model1.add(Dropout(0.2))
model1.add(Activation('tanh'))
#model1.add(LSTM(61,return_sequences=False))
#model1.add(Activation('tanh'))
model1.add(Dense(2, activation='sigmoid'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x_train, y_train, epochs=2, batch_size=1, verbose=2)


score = model1.evaluate(x_test, y_test, verbose=0)
y=model1.predict_proba(x_test)
j=metrics.roc_auc_score(y_test, y)
j=np.array(j)



model2=Sequential()
model2.add(LSTM(61,input_shape=(61, 3),return_sequences=False))
model2.add(Dropout(0.4))
model2.add(Activation('relu'))
#model1.add(LSTM(61,return_sequences=False))
#model1.add(Activation('tanh'))
model2.add(Dense(2, activation='softmax'))
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(x_train, y_train, epochs=2, batch_size=2, verbose=1)

score = model2.evaluate(x_test, y_test, verbose=0)
y=model2.predict_proba(x_test)
j=metrics.roc_auc_score(y_test, y)
j=np.array(j)

#0.7119437939110069


#saving your model
# serialize model to JSON
from keras.models import model_from_json
model2_json = model2.to_json()
import h5py
with open("model2.json", "w") as json_file:
    json_file.write(model2_json)
# serialize weights to HDF5
model.save_weights("model2.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")



#saving best modei and having that as checkpoint
# checkpoint
filepath="weights.best.hdf5"
heckpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)


# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(8, kernel_initializer='uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# load weights
model.load_weights("weights.best.hdf5")
# Compile model (required to make predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Created model and loaded weights from file")

model.add(LSTM(30,input_shape=(None,3)))
model.add(Dropout(0.2))
model.add(LSTM(10))