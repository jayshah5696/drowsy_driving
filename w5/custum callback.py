# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:36:43 2018

@author: Lenovo
"""

import keras
 
class My_Callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return  
from keras import callbacks

class roc(callbacks.Callback):
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
histories = my_callbacks.Histories()

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
		y_pred = self.model.predict(self.validation_data[0])
		self.aucs.append(roc_auc_score(self.validation_data[1], y_pred))
		return

	def on_batch_begin(self, batch, logs={}):
		return

	def on_batch_end(self, batch, logs={}):
		return
histories=Histories()


print(histories.losses)
print(histories.aucs)

import sklearn.metrics as sklm
class Metrics(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        #self.confusion = []
        #self.precision = []
        #self.recall = []
        #self.f1s = []
        #self.kappa = []
        self.auc = []

    def on_epoch_end(self, epoch, logs={}):
        score = np.asarray(self.model.predict(self.validation_data[0]))
        predict = np.round(np.asarray(self.model.predict(self.validation_data[0])))
        targ = self.validation_data[1]

        self.auc.append(sklm.roc_auc_score(targ, score))
        #self.confusion.append(sklm.confusion_matrix(targ, predict))
        #self.precision.append(sklm.precision_score(targ, predict))
        #self.recall.append(sklm.recall_score(targ, predict))
        #self.f1s.append(sklm.f1_score(targ, predict))
        #self.kappa.append(sklm.cohen_kappa_score(targ, predict))

        return

metrics=Metrics()

print(metrics.auc)

tensorboard = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,  
          write_graph=True, write_images=True)
filepath='E:\\USA\\Projects\\Research\\R_code\\w5\\weights\\weights.{epoch:02d}-{val_loss:.4f}.hdf5'
checkpoint=keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False,
                                           save_weights_only=False, mode='auto', period=1)


keras.callbacks.LambdaCallback( on_epoch_end= lambda )