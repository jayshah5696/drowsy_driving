library(h2o)
library(data.table)
library(plyr)
h2o.init(nthreads=-1,enable_assertions = FALSE)




colnames(train_balanced_scaled)[1]="td"
test_trimmed=test[,3:186]
test_trimmed$td=mapvalues(test_trimmed$td,c('TRUE','FALSE'),to=c('1','0'))
colnames(test_trimmed)=colnames(train_balanced_scaled)

train_balanced_scaled$td=as.factor(train_balanced_scaled$td)

test_trimmed_scaled=as.data.table(scale(test_trimmed[,2:184]))
test_trimmed_scaled=cbind(test_trimmed[,1],test_trimmed_scaled)
test_trimmed_scaled$td=as.factor(test_trimmed_scaled$td)
#write.csv(test_trimmed,"test_trimmed.csv")
#write.csv(test_trimmed_scaled,"test_trimmed_scaled.csv")
#write.csv(train_balanced_scaled,"train_balanced_scaled.csv")

datx = colnames(train_balanced_scaled[,2:184])
daty = colnames(train_balanced_scaled[,1])


trh=as.h2o(train_balanced_scaled,destination_frame = "trh")
tth=as.h2o(test_trimmed,destination_frame="tth")
txh=as.h2o(test_trimmed_scaled,destination_frame = "txh")

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(150,120,100,80,60,50,30,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=F,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)



plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


model_nn1 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn1",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(120,60,30,10), 
                              epochs = 100,
                              activation = "Tanh",standardize = F)
getwd()
h2o.saveModel(model_nn1, path="model_nn1", force = TRUE)
pretrained_model2=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'tth',
                                   hidden=c(120,60,30,10),
                                   epochs = 20,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn1',
                                   activation = 'Tanh',
                                   ignore_const_cols=F,standardize = F)


plot(pretrained_model2, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model2,train = T,valid = T)

#train     valid 
#0.9917280 0.7686703 
#both standardize =F
#train     valid 
#0.9999149 0.8181109 


model_nn2 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn2",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(150,120,90,60,30,10,6), 
                              epochs = 100,
                              activation = "Tanh",standardize = F,balance_classes = F)
getwd()
h2o.saveModel(model_nn2, path="model_nn2", force = TRUE)
pretrained_model1=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'tth',
                                   hidden=c(150,120,90,60,30,10,6),
                                   epochs = 20,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn2',
                                   activation = 'Tanh',
                                   ignore_const_cols=F)


plot(pretrained_model1, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model1,train = T,valid = T)
#train     valid 
#0.9987740 0.7210513 

model_nn3 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn3",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(120,60,30,10), 
                              epochs = 100,
                              activation = "Tanh",standardize = F)
getwd()
h2o.saveModel(model_nn1, path="model_nn1", force = TRUE)
pretrained_model2=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'tth',
                                   hidden=c(120,60,30,10),
                                   epochs = 20,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn1',
                                   activation = 'Tanh',
                                   ignore_const_cols=F,standardize = F)


plot(pretrained_model2, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model2,train = T,valid = T)

h2o.shutdown(prompt=F)


