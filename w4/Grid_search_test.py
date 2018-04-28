# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 12:40:23 2018

@author: Lenovo
"""
#change bins to 4 in filw to get 0.68 auc
runfile('E:\\USA\\Projects\\Research\\R_code\\w4\\start.py')

def cnnmodel(init='normal',act='relu',opti='adam',dr=0.0):
    model2=Sequential()
    model2.add(Convolution2D(61, (3,3), padding='same', input_shape=(61,61,1),
                         kernel_initializer=init,
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(Convolution2D(61, (3,3),kernel_initializer=init,
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(dr))

    model2.add(Convolution2D(32, (3,3), padding='same',kernel_initializer=init,
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(Convolution2D(32, (3, 3),kernel_initializer=init,
                         use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(dr))

    model2.add(Convolution2D(16, (3,3), padding='same'))
    model2.add(Activation(act))
    model2.add(Convolution2D(16, (3, 3)))
    model2.add(Activation(act))
    model2.add(MaxPooling2D(pool_size=(2,2)))
    model2.add(Dropout(dr))
    
    model2.add(Flatten())
    model2.add(Dense(256,kernel_initializer=init,
                             use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(Dense(64,kernel_initializer=init,
                             use_bias=True,bias_initializer='RandomNormal'))
    model2.add(Activation(act))
    model2.add(Dropout(dr))
    model2.add(Dense(1))
    model2.add(Activation('sigmoid'))
    
    model2.compile(loss='binary_crossentropy',metrics=[auc_roc,'accuracy'],optimizer=opti)
    #model2.summary()
    return model2
model = KerasClassifier(build_fn=cnnmodel, validation_data=(x_val, y_val)  verbose=0)####here look



init=['RandomNormal','RandomUniform','VarianceScaling','glorot_normal']  #kernerl Initializer
opti=['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax']
act=['relu','sigmoid','tanh']
dr=[0.0,0.25,0.5]
batch_size=[4,8,16]
epochs=[4,5,6,7,8]
n_iteration=1
scor={'roc_auc','accuracy'}
grid_params=dict(init=init,opti=opti,act=act,dr=dr,batch_size=batch_size,epochs=epochs)


grid=RandomizedSearchCV(estimator=model,param_distributions=grid_params,n_iter=n_iteration,scoring=scor,  
                        cv=5,refit='roc_auc',random_state=np.random.seed(seed),return_train_score=True)
#grid=RandomizedSearchCV(estimator=model,param_distributions=grid_params,n_iter=n_iteration,scoring=scor,  
 #                       pre_dispatch=2,iid=True,cv=10,refit='roc_auc',random_state=np.random.seed(seed),
  #                      n_jobs=-1, return_train_score=True)
  
idx = np.random.permutation(len(x_train))
x,y = x_train[idx], ytrain[idx]  
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
grid_result = grid.fit(x, encoded_Y,shuffle=True)

best_model=grid_result.best_estimator_
yhat=best_model.predict_proba(x_test)
print("auc: ",metrics.roc_auc_score(y_test, yhat))



print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    
    
for grid in grid_result.cv_results:
    print(grid.parameters)






grid=RandomizedSearchCV(estimator=model,param_distributions=grid_params,n_iter=n_iteration,scoring=scor,  
                        iid=True,cv=10,refit='roc_auc', n_jobs=2, return_train_score=True)
grid_result = grid.fit(x, encoded_Y)