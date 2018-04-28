library(data.table)
library(tidyverse)
library(dplyr)


#test

setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")
fullData = NULL
drives = list.files()
for(drive in drives){
  temp = fread(drive)[ ,c("Event_ID","RunName","TD","win60s","Time","Perclos")]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s,Event_ID)
  #temp$Time = round(temp$Time, 1)
  temp$nTime = floor(temp$Time*1)
  temp$drowsy = !is.na(temp$TD)
  temp = group_by(temp, RunName,win60s,Event_ID,nTime) %>% summarize(Perclos=median(Perclos, na.rm=T), drowsy=any(drowsy))
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  ##Now need to reshape
  if(max(temp$nTime) <= 60){
    #temp = spread(temp, key=nTime,value=Steering_Angle, sep="_")
    #temp = subset(temp, is.na(nTime_60) == F)
    #For Steering Angle Data only
    temp=temp[temp$Event_ID=='311',c("nTime","win60s","RunName","drowsy","Perclos")] 
    xmin=min(temp$win60s)
    temp=temp[temp$win60s!=xmin,]
    temp = spread(temp, key=nTime,value=Perclos, sep="_")
    temp = subset(temp, is.na(nTime_60) == F)
   
    fullData = rbind(fullData, temp)
  }
 
}


write.csv(fullData,"test_Perclos.csv")


#Train
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
fullData = NULL
drives = list.files()
for(drive in drives){
  temp = fread(drive)[ ,c("Event_ID","RunName","TD","win60s","Time","Perclos")]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s,Event_ID)
  #temp$Time = round(temp$Time, 1)
  temp$nTime = floor(temp$Time*1)
  temp$drowsy = !is.na(temp$TD)
  temp = group_by(temp, RunName,win60s,Event_ID,nTime) %>% summarize(Perclos=median(Perclos, na.rm=T), drowsy=any(drowsy))
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  ##Now need to reshape
  if(max(temp$nTime) <= 60){
    #temp = spread(temp, key=nTime,value=Steering_Angle, sep="_")
    #temp = subset(temp, is.na(nTime_60) == F)
    #For Steering Angle Data only
    temp=temp[temp$Event_ID=='311',c("nTime","win60s","RunName","drowsy","Perclos")] 
    xmin=min(temp$win60s)
    temp=temp[temp$win60s!=xmin,]
    temp = spread(temp, key=nTime,value=Perclos, sep="_")
    temp = subset(temp, is.na(nTime_60) == F)
    
    fullData = rbind(fullData, temp)
  }
  
}


write.csv(fullData,"train_Perclos.csv")



#Balancing
library(data.table)
library(OSTSC)


train=fread("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train_perclos.csv")
#test=fread("E:/USA/Projects/Research/Conferance/Individual_subjects/test_final.csv")
train=train[,-1]
#test=test[,-1]


test_balanced=OSTSC(sample=train[,4:64],label=train[,3])
x_train=as.data.table(test_balanced$sample)
colMeans(x_train[,5])
x_train_scale=scale(x_train)
x_train_scale=as.data.table(x_train_scale)
colMeans(x_train_scale)

y_train=as.data.table(test_balanced$label)
train_balanced_scaled=cbind(y_train,x_train_scale)
getwd()
write.csv(train_balanced_scaled,"train_balanced_scaled.csv")


#MLP model
test=fread("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test_perclos.csv")
test=test[,-1]
colnames(train_balanced_scaled)[1]="td"
test_trimmed=test[,3:64]

colnames(test_trimmed)=colnames(train_balanced_scaled)
test_trimmed$td=mapvalues(test_trimmed$td,c('TRUE','FALSE'),to=c('1','0'))

#reading your data
#
#
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/perclos")
train_balanced_scaled=fread("train_balanced_scaled.csv")
test_trimmed_scaled=fread("test_trimmed_scaled.csv")
#test_trimmed=fread("test_trimmed.csv")
test_trimmed=test_trimmed[,-1]
test_trimmed_scaled=test_trimmed_scaled[,-1]
train_balanced_scaled=train_balanced_scaled[,-1]


train_balanced_scaled$td=as.factor(train_balanced_scaled$td)
test_trimmed$td=as.factor(test_trimmed$td)
test_trimmed_scaled=as.data.table(scale(test_trimmed[,2:62]))
test_trimmed_scaled=cbind(test_trimmed[,1],test_trimmed_scaled)
test_trimmed_scaled$td=as.factor(test_trimmed_scaled$td)
#write.csv(test_trimmed,"test_trimmed.csv")
#write.csv(test_trimmed_scaled,"test_trimmed_scaled.csv")
#write.csv(train_balanced_scaled,"train_balanced_scaled.csv")

datx = colnames(train_balanced_scaled[,2:62])
daty = colnames(train_balanced_scaled[,1])

library(h2o)
h2o.init(nthreads = -1)

trh=as.h2o(train_balanced_scaled,destination_frame = "trh")
tth=as.h2o(test_trimmed,destination_frame="tth")
txh=as.h2o(test_trimmed_scaled,destination_frame = "txh")

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(64,64,64,64), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=F,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'txh',
                            l1=0.00158)
plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)



model1 <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(32,32,32,32), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=F,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'txh',
                            l1=0.00158)
plot(model1, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model1,train = T,valid = T)


model_nn1 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn1",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(50,40,30,20,10), 
                              epochs = 100,
                              activation = "Tanh",standardize = F)
getwd()
get=h2o.saveModel(model_nn1, path="model_nn2", force = TRUE)
pretrained_model2=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'txh',
                                   hidden=c(50,40,30,20,10),
                                   epochs = 15,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn1',
                                   activation = 'Tanh',
                                   ignore_const_cols=F,reproducible = T)
plot(pretrained_model2, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model2,train = T,valid = T)
h2o.saveModel(pretrained_model2, path="best_model_0.60_for_perclos", force = TRUE)

saved_model=h2o.loadModel("E:\\USA\\Projects\\Research\\R_code\\w2\\best_model_0.60_for_perclos\\DeepLearning_model_R_1517697720271_8000")
prob=h2o.predict(saved_model,txh)
df=as.data.frame(prob)
roc_auc(df[,3],test_trimmed_scaled[,1])
mlp_perclos=cbind(test_trimmed_scaled$td,df)
write.csv(mlp_perclos,"mlp_perclos.csv")


getwd()
#1 for 20 epochs 
#
#
#  train     valid 
#0.6512767 0.5110591
#
#2 for 50 epochs
#train     valid 
#0.8381804 0.5862607 
#3  for 100 epochs
# train     valid 
#0.7833143 0.6029144
#4   train     valid 
#0.8405361 0.6710903 
#
saved_model=h2o.loadModel(model_nn1)

model_nn2 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn2",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(32,32,32,32,32), 
                              epochs = 100,
                              activation = "Tanh",standardize = F)
getwd()
h2o.saveModel(model_nn1, path="model_nn1", force = TRUE)
pretrained_model3=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'tth',
                                   hidden=c(32,32,32,32,32),
                                   epochs = 20,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn2',
                                   activation = 'Tanh',
                                   ignore_const_cols=F)
plot(pretrained_model2, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model2,train = T,valid = T)
                                   

model_nn3 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn3",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(32,32,32,32,32), 
                              epochs = 100,
                              activation = "Tanh",standardize = F,initial_weight_distribution = "Normal")
getwd()
h2o.saveModel(model_nn3, path="model_nn3", force = TRUE)
pretrained_model4=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'txh',
                                   hidden=c(32,32,32,32,32),
                                   epochs = 200,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn3',
                                   activation = 'Tanh',distribution = "bernoulli",
                                   ignore_const_cols=F,standardize = F,l1=0.00156)
plot(pretrained_model4, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model4,train = T,valid = T)
#train    valid 
#0.340866 0.568306

model_nn4 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn4",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(32,16,8), 
                              epochs = 100,
                              activation = "Tanh",standardize = F,initial_weight_distribution = "Normal")
getwd()
h2o.saveModel(model_nn4, path="model_nn4", force = TRUE)
pretrained_model5=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'txh',
                                   hidden=c(32,16,8),
                                   epochs = 200,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn4',
                                   activation = 'Tanh',distribution = "bernoulli",
                                   ignore_const_cols=F,standardize = F,l1=0.00156)
plot(pretrained_model5, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model5,train = T,valid = T)
#train     valid 
#0.4340521 0.5597190 
#
model_nn5 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn5",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(32,32,32,32,32), 
                              epochs = 100,
                              activation = "Tanh",standardize = F,initial_weight_distribution = "Normal")
getwd()
h2o.saveModel(model_nn3, path="model_nn5", force = TRUE)
pretrained_model6=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'tth',
                                   hidden=c(32,32,32,32,32),
                                   epochs = 200,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn5',
                                   activation = 'Tanh',distribution = "bernoulli",
                                   ignore_const_cols=F,standardize = F,l1=0.00156,reproducible = T)
plot(pretrained_model6, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model6,train = T,valid = T)
#train     valid 
#0.8016864 0.5896435
#
model_nn6 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn6",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(32,32,32,32,32), 
                              epochs = 100,
                              activation = "Tanh",standardize = F,initial_weight_distribution = "Normal")
#getwd()
#h2o.saveModel(model_nn3, path="model_nn5", force = TRUE)
pretrained_model7=h2o.deeplearning(x=datx,y=daty,
                                   training_frame = 'trh',
                                   validation_frame = 'txh',
                                   hidden=c(32,32,32,32,32),
                                   epochs = 200,
                                   stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                   seed=1001,pretrained_autoencoder = 'model_nn6',
                                   activation = 'Tanh',distribution = "bernoulli",
                                   ignore_const_cols=F,standardize = F,l1=0.00156,reproducible = T)
plot(pretrained_model7, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model7,train = T,valid = T)
#   train     valid 
#0.8575363 0.6541764 
