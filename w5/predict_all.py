# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 15:53:44 2018

@author: Lenovo
"""

runfile('E:\\USA\\Projects\\Research\\R_code\\w4\\start.py')


#using validation set approch

#idx = np.random.permutation(len(x_train))
#x,y = x_train[idx], y_train[idx]

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



model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=8, epochs=10, verbose=1,callbacks=[histories])
model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=8, epochs=1, verbose=1)
#model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=16, epochs=1, verbose=1)


#model2.fit(x, y, validation_data=(x_test, y_test),batch_size=8, epochs=6, verbose=1)
#model2.fit(x, y, validation_data=(x_test, y_test),batch_size=16, epochs=5, verbose=1)

#yhat=model2.predict_proba(x_train)
#print("auc: ",metrics.roc_auc_score(y_train, yhat))


yhat=model2.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(y_test, yhat))

#roc_mtf=np.column_stack((y_test,yhat))
#roc_df=pd.DataFrame(roc_mtf)
#roc_df.to_csv("roc_val_0.75_train_0.82.csv")



model2.save("E:\\USA\\Projects\\Research\\R_code\\w5\\my_model_0.7431.h5")


model2.save_weights("E:\\USA\\Projects\\Research\\R_code\\w5\\model_weight_0.74.h5")
loaded_model.load_weights("E:\\USA\\Projects\\Research\\R_code\\w5\\model_weight_0.74.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy',metrics=['accuracy'],optimizer='Adadelta')
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


scores = model2.evaluate(x_test, y_test, verbose=0)
print("%s: %.2f%%" % (model2.metrics_names[1], scores[1]*100))














runfile('E:\\USA\\Projects\\Research\\R_code\\w4\\start.py')


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



model2.fit(x_train, y_train, validation_data=(x_test, y_test),batch_size=8, epochs=50, verbose=1,callbacks=[histories,tensorboard,checkpoint])


yhat=model2.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(y_test, yhat))

print(histories.aucs)
auc=pd.DataFrame(histories.aucs)
auc.to_csv("E:\\USA\\Projects\\Research\\R_code\\w5\\AUC.csv")
#30 ,34
#Run following command in terminal ruf
#tensorboard --logdir E:/USA/Projects/Research/R_code/w2/Graph 

#loading weights from saved model
load=load_model(filepath="E:\\USA\\Projects\\Research\\R_code\\w5\\weights\\weights.30-0.62.hdf5",
                custom_objects={'auc_roc':auc_roc})
#checking the model
yhat=load.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(y_test, yhat))

yhat=load.predict_proba(x_train_all)
print("auc: ",metrics.roc_auc_score(y_train_all, yhat))
#predicting on the data set with all other events

roc_train_all=np.column_stack((y_train_all,yhat))
roc_df=pd.DataFrame(roc_train_all)
roc_df.to_csv("E:\\USA\\Projects\\Research\\R_code\\w5\\roc_train_all.csv")