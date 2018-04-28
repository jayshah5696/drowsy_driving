library(data.table)
t1=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_30_steer_label.csv')
test1=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_30_steer_label.csv')
t1=t1[,-1]
test1=test1[,-1]
d=c(1:31)
d=as.character(d)
colnames(t1)=d
colnames(test1)=d
t2=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_30_break_label.csv')
test2=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_30_break_label.csv')
t2=t2[,-1]
test2=test2[,-1]
d=c(32:36)
d=as.character(d)
colnames(t2)=d
colnames(test2)=d
t3=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_30_acc_label.csv')
test3=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_30_acc_label.csv')
t3=t3[,-1]
test3=test3[,-1]
d=c(37:45)
d=as.character(d)
colnames(t3)=d
colnames(test3)=d


train=cbind(t3,t2[,4:5],t1[,4:31])
test=cbind(test3,test2[,4:5],test1[,4:31])
colnames(train)[2]='td'
colnames(test)=colnames(train)
library(h2o)
h2o.init()

train$td=as.factor(train$td)
test$td=as.factor(test$td)


datx = colnames(train[,4:39])
daty = colnames(train[,2])
trh=as.h2o(train,destination_frame = "trh")
tth=as.h2o(test,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(20,20,20), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)
plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


t1=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_60_steer_label.csv')
test1=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_60_steer_label.csv')
t1=t1[,-1]
test1=test1[,-1]

t2=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_60_break_label.csv')
test2=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_60_break_label.csv')
t2=t2[,-1]
test2=test2[,-1]

t3=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_60_acc_label.csv')
test3=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_60_acc_label.csv')
t3=t3[,-1]
test3=test3[,-1]



train=cbind(t3,t2[,4:5],t1[,4:58])
test=cbind(test3,test2[,4:5],test1[,4:58])
d=c(1:73)
d=as.character(d)
colnames(train)=d
colnames(train)[2]='td'
colnames(test)=colnames(train)


train$td=as.factor(train$td)
test$td=as.factor(test$td)


datx = colnames(train[,4:73])
daty = colnames(train[,2])
trh=as.h2o(train,destination_frame = "trh")
tth=as.h2o(test,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(20,20,20), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            l1=0.00158,validation_frame = 'tth')
plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)
