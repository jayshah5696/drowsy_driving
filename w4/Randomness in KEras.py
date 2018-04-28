# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 05:24:28 2018

@author: Lenovo
"""
from pandas import DataFrame
from pandas import concat
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error
import numpy as np
from tensorflow import set_random_seed
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#seeding 1001
seed = 1001
np.random.seed(10) #for numpy
set_random_seed(10) #for Tensorflow  



# fit MLP to dataset and print error
def fit_model(X, y):
	# design network
	model = Sequential()
	model.add(Dense(10, input_dim=1))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	model.fit(X, y, epochs=100, batch_size=len(X), verbose=0)
	# forecast
	yhat = model.predict(X, verbose=0)
	print(mean_squared_error(y, yhat[:,0]))
 
# create sequence
length = 10
sequence = [i/float(length) for i in range(length)]
# create X/y pairs
df = DataFrame(sequence)
df = concat([df.shift(1), df], axis=1)
df.dropna(inplace=True)
# convert to MLP friendly format
values = df.values
X, y = values[:,0], values[:,1]
# repeat experiment

runfile('E:\\USA\\Projects\\Research\\R_code\\w4\\Reproducible.py')
                                                                                                                                                                                repeats = 10
 for _ in range(repeats):                                                                                                                                                                             	fit_model(X, y)
 
 
 
 
 #Kfold Cross Validation


import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(60, input_dim=60, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model


def cnnmodel():
    model2=Sequential()
    model2.add(Convolution2D(61, (3,3), padding='same', input_shape=(61,61,1),
                         kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(Convolution2D(61, (3,3),kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(0.25))

    model2.add(Convolution2D(32, (3,3), padding='same',kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(Convolution2D(32, (3, 3),kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(0.25))

    model2.add(Convolution2D(16, (3,3), padding='same'))
    model2.add(Activation('relu'))
    model2.add(Convolution2D(16, (3, 3)))
    model2.add(Activation('relu'))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(0.25))
    
    model2.add(Flatten())
    model2.add(Dense(256,kernel_initializer='normal',
                             use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(Dense(64,kernel_initializer='normal',
                             use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation('relu'))
    model2.add(Dropout(0.5))
    model2.add(Dense(1))
    model2.add(Activation('sigmoid'))
    
    model2.compile(loss='binary_crossentropy',metrics=[auc_roc,'accuracy'],optimizer='Adadelta')
    #model2.summary()
    return model2

#model2.fit(x, y, batch_size=8, epochs=6, verbose=1)
# evaluate model with standardized dataset
    
# encode class values as integers
x=x_train
encoder = LabelEncoder()
encoder.fit(ytrain)
encoded_Y = encoder.transform(ytrain)
estimator = KerasClassifier(build_fn=cnnmodel, epochs=5, batch_size=8, verbose=0)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
results = cross_val_score(estimator, x, encoded_Y, cv=kfold,scoring='roc_auc')
results
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

?cross_val_score


from sklearn import metrics
roc_auc_score






#custom checkpoint for auc roc
import keras
from sklearn.metrics import roc_auc_score
 
class Histories(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        y_pred = self.model.predict(self.model.validation_data[0])
        self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return
#prepare callback before fitting
histories = my_callbacks.Histories()

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=[histories])
print(histories.losses)
print(histories.aucs)

from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt




#CHECKPOINTING
filepath="weights-improvement-{epoch:02d}-{val_auc_roc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_auc_roc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)





















#using validation set approch



model2=Sequential()
model2.add(Convolution2D(61, (3,3), padding='same', input_shape=(61,61,1),
                         kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Convolution2D(61, (3,3),kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

model2.add(Convolution2D(32, (3,3), padding='same',kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Convolution2D(32, (3, 3),kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

model2.add(Convolution2D(16, (3,3), padding='same'))
model2.add(Activation('relu'))
model2.add(Convolution2D(16, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))

model2.add(Flatten())
model2.add(Dense(256,kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Dense(64,kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(2))
model2.add(Activation('softmax'))

model2.compile(loss='binary_crossentropy',metrics=[auc_roc,'accuracy'],optimizer='Adadelta')
#model2.summary()

model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=8, epochs=6, verbose=1)
model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=16, epochs=1, verbose=1)

yhat=model2.predict_proba(x_train)
print("auc: ",metrics.roc_auc_score(y_train, yhat))


yhat=model2.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(y_test, yhat))

roc_mtf=np.column_stack((y_test,yhat))
roc_df=pd.DataFrame(roc_mtf)
roc_df.to_csv("roc_val_0.75_train_0.82.csv")



#manual k fold cv
encoder = LabelEncoder()
encoder.fit(ytrain)
encoded_Y = encoder.transform(ytrain)
# define 10-fold cross validation test harness

   
















model2=Sequential()
model2.add(Convolution2D(61, (3,3), padding='same', input_shape=(61,61,1),
                    kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Convolution2D(61, (3,3),kernel_initializer='normal',
                         use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
 
model2.add(Convolution2D(32, (3,3), padding='same',kernel_initializer='normal',
                       use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Convolution2D(32, (3, 3),kernel_initializer='normal',
                            use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
    
model2.add(Convolution2D(16, (3,3), padding='same'))
model2.add(Activation('relu'))
model2.add(Convolution2D(16, (3, 3)))
model2.add(Activation('relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
    
model2.add(Flatten())
model2.add(Dense(256,kernel_initializer='normal',
                             use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Dense(64,kernel_initializer='normal',
                             use_bias=True,bias_initializer='RandomNormal'))
model2.add(Activation('relu'))
model2.add(Dropout(0.5))
model2.add(Dense(1))
model2.add(Activation('sigmoid'))
#Compile model
model2.compile(loss='binary_crossentropy',metrics=[auc_roc,'accuracy'],optimizer='Adadelta')






kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(x, encoded_Y):
  # create model
 
	# Fit the model
    model2.fit(x[train], encoded_Y[train], batch_size=8, epochs=6, verbose=0,initial_epoch=0)
    scores = model2.evaluate(x[test], encoded_Y[test], verbose=1)
    print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    yhat=model2.predict_proba(x[test])
    print("auc: ",metrics.roc_auc_score(encoded_Y[test], yhat))
    y_pred = yhat[:,0] > 0.5
    #print(metrics.classification_report(encoded_Y[test][:,0],y_pred))
    
    
x=x_train
encoder = LabelEncoder()
encoder.fit(ytrain)
encoded_Y = encoder.transform(ytrain)    
model2.fit(x,encoded_Y,batch_size=8,epochs=6,verbose=1,initial_epoch=0)
#model2.fit(x,encoded_Y,batch_size=8,epochs=2,verbose=1)
yhat=model2.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(ytest, yhat))


ytest2=ytest.astype(np.float32)
#roc_mtf=np.concatenate((ytest2,yhat),axis=1)
roc_mtf=np.column_stack((ytest,yhat))
np.savetxt("roc_mtf.csv", roc_mtf, delimiter=",")



from keras.models import load_model

model2.save('my_model.h5') 