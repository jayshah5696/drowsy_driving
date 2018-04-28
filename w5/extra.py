# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 21:11:49 2018

@author: Lenovo
"""

import pandas as pd

train =pd.read_csv('train_balanced.csv')
test=pd.read_csv('train_balanced_scaled.csv')
train=train.drop(train.columns[[0]], axis=1)
test=test.drop(test.columns[[0]],axis=1)

xtrain=train.iloc[:,0:61]
xtrain=xtrain.values
xtrain.shape
ytrain=test.iloc[:,1]
ytrain=ytrain.values



xtest=test.iloc[:,3:64]
xtest=xtest.values
ytest=test.iloc[:,2]
ytest=ytest.values


from keras.utils import to_categorical
y_train = to_categorical(ytrain)
y_test=to_categorical(ytest)



from pyts.transformation import StandardScaler

standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(xtrain)
Xt_standardized = standardscaler.transform(xtest)

from pyts.transformation import GASF, GADF
from pyts import transformation, classification, visualization

gasf = GASF(image_size=61, overlapping=False, scale='-1')
X_gasf = gasf.transform(X_standardized)
X_gasf.ndim
#3
Xt_gasf=gasf.transform(Xt_standardized)


gadf = GADF(image_size=61, overlapping=False, scale='-1')
X_gadf = gadf.transform(X_standardized)
Xt_gadf = gadf.transform(Xt_standardized)


from pyts.visualization import plot_gasf
plot_gasf(X_standardized[0], image_size=61, overlapping=False, scale='-1')

from pyts.visualization import plot_gadf
plot_gadf(X_standardized[4], image_size=30, overlapping=False, scale='-1')



x_train=X_gadf.reshape
x_train =X_gadf.reshape(X_gadf.shape[0],61,61,1)
x_test =Xt_gadf.reshape(Xt_gadf.shape[0],61,61,1)
#-----------------------------------------------------------------------------------------------------------------------------------------------------


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


model = Sequential()
model.add(Convolution2D(61, kernel_size=(10, 10), strides=(1, 1),
                 activation='relu',
                 input_shape=(61,61,1)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Convolution2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=[auc_roc,'accuracy'])


model.fit(x, y,batch_size=4, nb_epoch=2, verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
from sklearn import  metrics
y=model.predict_proba(x_test)
j=metrics.roc_auc_score(y_test, y)
j=np.array(j)
print(j)