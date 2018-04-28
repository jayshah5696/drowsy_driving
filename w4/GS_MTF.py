# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 17:41:34 2018

@author: Lenovo
"""
#Setting Up directory
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
#from pandas_ml import ConfusionMatrix
from tensorflow import set_random_seed
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
xtrain=xtrain.values
ytrain=train.iloc[:,0]
ytrain=ytrain.values

#loading Test data
xtest=test.iloc[:,3:64]
xtest=xtest.values
ytest=test.iloc[:,2]
ytest=ytest.values


y_train = to_categorical(ytrain)
y_test=to_categorical(ytest)

#reshaping time series window
standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(xtrain)
Xt_standardized = standardscaler.transform(xtest)





#defining custome metric for guiding our neural network
def auc_roc(y_true, y_pred):
    # any tensorflow metric
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









##########################################################################################
#building model for mtf transformation
mtf = MTF(image_size=61, n_bins=4, quantiles='empirical', overlapping=False)
X_mtf = mtf.transform(X_standardized)
Xt_mtf=mtf.transform(Xt_standardized)

x_train =X_mtf.reshape(X_mtf.shape[0],61,61,1)
x_test =Xt_mtf.reshape(Xt_mtf.shape[0],61,61,1)

#shuffling the data set

idx = np.random.permutation(len(x_train))
x,y = x_train[idx], y_train[idx]

###########################################################################

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
model2.summary()

model2.fit(x, y, batch_size=8, epochs=6, verbose=1)



yhat=model2.predict_proba(x_test)
auc=metrics.roc_auc_score(y_test, yhat)
#y_true = np.array([0] * 1000 + [1] * 1000)
y_pred = yhat[:,0] > 0.5
metrics.confusion_matrix(y_test[:,0], y_pred)
auc=np.array(auc)
print("auc: ",metrics.roc_auc_score(y_test, yhat))

report=metrics.classification_report(y_test[:,0],y_pred)
print(report)
#cm = ConfusionMatrix(y_test[:,0], y_pred)
#cm.print_stats()





def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

plot_confusion_matrix(df_confusion)












model = Sequential()
model.add(Convolution2D(61, kernel_size=(10, 10), strides=(1,1),activation='relu',
                        input_shape=(61,61,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(61, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))



#Grid Search
#building sequential model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Convolution2D(61, kernel_size=(10, 10),
                        strides=(1,1),activation='relu',input_shape=(61,61,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Convolution2D(61, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=[auc_roc])
    return model
#model.summary()

#model.fit(x, y, batch_size=batch_size, nb_epoch=epochs, verbose=1)
model = KerasClassifier(build_fn=create_model,   verbose=1)




#applying Grid Search
# define the grid search parameters
batch_size = [4,8,16]
epochs = [1,2,3,4,5]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer)
grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=1,                          n_iter=1,cv=10,random_state=np.random.seed(seed))
grid_result = grid.fit(x, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))



#predicting
score = model.evaluate(x_test, y_test, verbose=1)
yhat=model.predict_proba(x_test)
j=metrics.roc_auc_score(y_test, yhat)
j=np.array(j)
print(j)

roc_mtf=np.concatenate((y_test,yhat),axis=1)
np.savetxt("roc_mtf.csv", roc_mtf, delimiter=",")
