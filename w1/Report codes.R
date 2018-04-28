#here we are gonna focus on first only pca data
#and only on steering data
#1) 10 s data

library(data.table)
library(h2o)
h2o.init(nthreads = -1)

h2o.removeAll()
train=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_10_steer_label.csv')
test=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_10_steer_label.csv')
train=train[,-1]
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)


datx = colnames(train[,4:14])
daty = colnames(train[,2])
trh=as.h2o(train,destination_frame = "trh")
tth=as.h2o(test,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
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


#without pca
train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/train.csv")
train=train[,-1]
test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/test.csv")
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)



#selecting only steering data
train1=train[,c(1:3,3604:5403)]
test1=test[,c(1:3,3604:5403)]



datx = colnames(train1[,4:1803])
daty = colnames(train1[,2])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")

#getting feel of model on data

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.5,0.5,.5,0.5,0.5,0.5), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(600,600,300,300,300,300), # three layers of 50 nodes
                            epochs = 1, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)
,epsilon=10e-4)
,
rho=0.999,epsilon=10e-8


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#Applying balancing with OSTSC package

library(OSTSC)
balanced=OSTSC(train1[,4:603],train1[,2],parallel=T,progBar = T)
#taking too much time

library(DMwR)
table(train1$td)
balanced_data=SMOTE(td~.,train1[,c(2,4:603)],perc.over=5000,perc.under=200,k=10)
table(balanced_data$td)


trh=as.h2o(balanced_data,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#applying weighted matrix
weight=read.csv('je.csv')
weight[,3]=as.numeric(weight[,3])
train1=cbind(train1,weight[,3])

colnames(train1)[605]="weight"
colnames(train1[,605])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8,
                            weights_column = 'weigh')

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2)30s data
#pcaed data
train=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_30_steer_label.csv')
test=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_30_steer_label.csv')
train=train[,-1]
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)


datx = colnames(train[,4:31])
daty = colnames(train[,2])
trh=as.h2o(train,destination_frame = "trh")
tth=as.h2o(test,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")


#without pca
train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/train.csv")
train=train[,-1]
test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/test.csv")
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)



#selecting only steering data
train1=train[,c(1:3,3604:5403)]
test1=test[,c(1:3,3604:5403)]



datx = colnames(train1[,4:1803])
daty = colnames(train1[,2])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")

#getting feel of model on data

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)




model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(150,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)






model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(150,100,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8)



plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#Applying balancing with OSTSC package

library(OSTSC)
balanced=OSTSC(train1[,4:603],train1[,2],parallel=T,progBar = T)
#taking too much time

library(DMwR)
table(train1$td)
balanced_data=SMOTE(td~.,train1[,c(2,4:1803)],perc.over=5000,perc.under=200,k=10)
table(balanced_data$td)


trh=as.h2o(balanced_data,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(50,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#applying weighted matrix
weight=read.csv('je.csv')
weight[,3]=as.numeric(weight[,3])
train1=cbind(train1,weight[,3])

colnames(train1)[605]="weight"
colnames(train1[,605])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.5,0.5,.5,0.5,0.5), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,300,300,300,300), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3)60s data
#pcaed data
train=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/train_prdata_60_steer_label.csv')
test=fread('E:/USA/Projects/Research/DATASET/Individual_subjects/test_prdata_60_steer_label.csv')
train=train[,-1]
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)


datx = colnames(train[,4:58])
daty = colnames(train[,2])
trh=as.h2o(train,destination_frame = "trh")
tth=as.h2o(test,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
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

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)

#without pca
train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/60s/train.csv")
train=train[,-1]
test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/60s/test.csv")
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)



#selecting only steering data
train1=train[,c(1:3,7204:10803)]
test1=test[,c(1:3,7204:10803)]



datx = colnames(train1[,4:3603])
daty = colnames(train1[,2])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")

#getting feel of model on data

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(50,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)




model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(150,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)






model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(150,100,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8)



plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#Applying balancing with OSTSC package

library(OSTSC)
balanced=OSTSC(train1[,4:603],train1[,2],parallel=T,progBar = T)
#taking too much time

library(DMwR)
table(train1$td)
balanced_data=SMOTE(td~.,train1[,c(2,4:1803)],perc.over=5000,perc.under=200,k=10)
table(balanced_data$td)


trh=as.h2o(balanced_data,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(50,50,50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.4,0.4,.5,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,150,50,20,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,
                            rho=0.999,epsilon=10e-8)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


#applying weighted matrix
weight=read.csv('je.csv')
weight[,3]=as.numeric(weight[,3])
train1=cbind(train1,weight[,3])

colnames(train1)[605]="weight"
colnames(train1[,605])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.5,0.5,.5,0.5,0.5), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(300,300,300,300,300), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)










h2o.shutdown(prompt=FALSE)
