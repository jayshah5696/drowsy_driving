# -*- coding: utf-8 -*-
"""
Created on Sun Feb 25 16:11:50 2018

@author: Lenovo
"""
#This is for the data without 311 cases
train_all =pd.read_csv("E:\\USA\\Projects\\Research\\R_code\w5\\weights\\train_all.csv")
#train_all =pd.read_csv("E:\\USA\\Projects\\Research\\R_code\w5\\balanced_all.csv")
test_all=pd.read_csv('E:\\USA\\Projects\\Research\\R_code\\w5\\weights\\test_all.csv')
#train_all=train.drop(train_all.columns[0], axis=1)
#test_all=test.drop(test_all.columns[0],axis=1)

xtrain_all=train_all.iloc[:,4:65]
xtrain_all=np.array(xtrain_all.values)

ytrain_all=train_all.iloc[:,3]
ytrain_all=np.array(ytrain_all.values)


#xtrain_all=train_all.iloc[:,1:62]
#xtrain_all=np.array(xtrain_all.values)
#ytrain_all=train_all.iloc[:,62]
#ytrain_all=np.array(ytrain_all.values)

#loading Test data
xtest_all=test_all.iloc[:,4:65]
xtest_all=np.array(xtest_all.values)
ytest_all=test_all.iloc[:,3]
ytest_all=np.array(ytest_all.values)


y_train_all = to_categorical(ytrain_all)
y_test_all=to_categorical(ytest_all)

#reshaping time series window
standardscaler = StandardScaler(epsilon=1e-2)
X_standardized_all = standardscaler.transform(xtrain_all)
Xt_standardized_all = standardscaler.transform(xtest_all)


mtf = MTF(image_size=61, n_bins=6, quantiles='empirical', overlapping=False)
X_mtf_all = mtf.transform(X_standardized_all)
Xt_mtf_all=mtf.transform(Xt_standardized_all)

x_train_all =X_mtf_all.reshape(X_mtf_all.shape[0],61,61,1)
x_test_all =Xt_mtf_all.reshape(Xt_mtf_all.shape[0],61,61,1)


train_all[['RunName','win60s']]



#This is for the all cases
train_with_all_events =pd.read_csv('E:\USA\Projects\Research\R_code\w5\train_with_all_events.csv')
test_with_all_events=pd.read_csv('E:\USA\Projects\Research\R_code\w5\test_with_all_events.csv')
train_with_all_events=train.drop(train_with_all_events.columns[[0]], axis=1)
test_with_all_events=test.drop(test_with_all_events.columns[[0]],axis=1)


xtrain_with_all_events=train_with_all_events.iloc[:,3:64]
xtrain_with_all_events=np.array(xtrain_with_all_events.values)
ytrain_with_all_events=train_with_all_events.iloc[:,2]
ytrain_with_all_events=np.array(ytrain_with_all_events.values)

#loading Test data
xtest_with_all_events=test_with_all_events.iloc[:,3:64]
xtest_with_all_events=np.array(xtest_with_all_events.values)
ytest_with_all_events=test_with_all_events.iloc[:,2]
ytest_with_all_events=np.array(ytest_with_all_events.values)


y_train_with_all_events = to_categorical(ytrain_with_all_events)
y_test_with_all_events=to_categorical(ytest_with_all_events)

#reshaping time series window
standardscaler = StandardScaler(epsilon=1e-2)
X_standardized_with_all_events = standardscaler.transform(xtrain_with_all_events)
Xt_standardized_with_all_events = standardscaler.transform(xtest_with_all_events)


mtf = MTF(image_size=61, n_bins=6, quantiles='empirical', overlapping=False)
X_mtf_with_all_events = mtf.transform(X_standardized_with_all_events)
Xt_mtf_with_all_events=mtf.transform(Xt_standardized_with_all_events)

x_train_with_all_events =X_mtf_with_all_events.reshape(X_mtf_with_all_events.shape[0],61,61,1)
x_test_with_all_events =Xt_mtf_with_all_events.reshape(Xt_mtf_with_all_events.shape[0],61,61,1)

