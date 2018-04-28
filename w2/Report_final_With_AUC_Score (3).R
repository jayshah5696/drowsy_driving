library(data.table)
library(tidyverse)
library(dplyr)


#getting dataset
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")

fullData = NULL  #Creating a Data frame
drives = list.files() #getting list of files in working Directory
for(drive in drives){#creating for loop in the wd 
  temp = fread(drive)[,c("Event_ID","RunName","TD","win60s","Time","Steering_Angle","Brake_Pedal_Pos","Acceler_Pedal_Pos")] # reading only 5 Columns from Individual Dataset
  #temp=temp[temp$Event_ID=='311',]
  #xmax=min(temp$win60s)
  #temp=temp[temp$win60s!=xmax]
  temp = arrange(temp, Time)#arranging the dataframe by time column
  temp = group_by(temp, RunName, win60s,Event_ID) #setting up data with grouping time , RunName, win 60s columns
  #  temp$nTime = round(temp$Time, 1)
  temp$nTime = round(temp$Time*1) / 2 #creating a reduced frequency time vector with 2HZ
  temp$drowsy = !is.na(temp$TD) #creating a Drowsy Column with True and False
  temp = group_by(temp, RunName,win60s,Event_ID,nTime) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy), Brake_Pedal_Pos=median(Brake_Pedal_Pos, na.rm=T), Acceler_Pedal_Pos=median(Acceler_Pedal_Pos, na.rm=T))
  #Reducing Frequency of Data(SteerIng_Angle)
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  #Computes and adds new variable(s). Preserves existing variables. It's similar to the R base function transform().transmute(): Computes new variable(s). Drops existing variables.
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  ##Now need to reshape
  if(max(temp$nTime) <= 60){
    
    
    temp1=temp[temp$Event_ID=='311',c("nTime","win60s","RunName","drowsy","Steering_Angle")] 
    xmin=min(temp1$win60s)
    temp1=temp1[temp1$win60s!=xmin,]
    temp1 = spread(temp1, key=nTime,value=Steering_Angle, sep="_")
    temp1 = subset(temp1, is.na(nTime_60) == F)
    temp2=temp[,c("nTime","win60s","RunName","Acceler_Pedal_Pos")] 
    temp2 = spread(temp2, key=nTime,value=Acceler_Pedal_Pos, sep="_")
    temp3=temp[,c("nTime","win60s","RunName","Brake_Pedal_Pos")] 
    temp3 = spread(temp3, key=nTime,value=Brake_Pedal_Pos, sep="_")
    temp2 = subset(temp2, is.na(nTime_60) == F)
    temp3 = subset(temp3, is.na(nTime_60) == F)
    temp=cbind(temp1,temp2,temp3)
    
    fullData = rbind(fullData, temp)
  }
}
write.csv(fullData,"TEst_sir.csv")

setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")

fullData = NULL  #Creating a Data frame
drives = list.files() #getting list of files in working Directory
for(drive in drives){#creating for loop in the wd 
  temp = fread(drive)[,c("Event_ID","RunName","TD","win60s","Time","Steering_Angle","Brake_Pedal_Pos","Acceler_Pedal_Pos")] # reading only 5 Columns from Individual Dataset
  #temp=temp[temp$Event_ID=='311',]
  #xmax=min(temp$win60s)
  #temp=temp[temp$win60s!=xmax]
  temp = arrange(temp, Time)#arranging the dataframe by time column
  temp = group_by(temp, RunName, win60s,Event_ID) #setting up data with grouping time , RunName, win 60s columns
  #  temp$nTime = round(temp$Time, 1)
  temp$nTime = round(temp$Time*1) / 2 #creating a reduced frequency time vector with 2HZ
  temp$drowsy = !is.na(temp$TD) #creating a Drowsy Column with True and False
  temp = group_by(temp, RunName,win60s,Event_ID,nTime) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy), Brake_Pedal_Pos=median(Brake_Pedal_Pos, na.rm=T), Acceler_Pedal_Pos=median(Acceler_Pedal_Pos, na.rm=T))
  #Reducing Frequency of Data(SteerIng_Angle)
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  #Computes and adds new variable(s). Preserves existing variables. It's similar to the R base function transform().transmute(): Computes new variable(s). Drops existing variables.
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  ##Now need to reshape
  if(max(temp$nTime) <= 60){
    
    
    temp1=temp[temp$Event_ID=='311',c("nTime","win60s","RunName","drowsy","Steering_Angle")] 
    xmin=min(temp1$win60s)
    temp1=temp1[temp1$win60s!=xmin,]
    temp1 = spread(temp1, key=nTime,value=Steering_Angle, sep="_")
    temp1 = subset(temp1, is.na(nTime_60) == F)
    temp2=temp[,c("nTime","win60s","RunName","Acceler_Pedal_Pos")] 
    temp2 = spread(temp2, key=nTime,value=Acceler_Pedal_Pos, sep="_")
    temp3=temp[,c("nTime","win60s","RunName","Brake_Pedal_Pos")] 
    temp3 = spread(temp3, key=nTime,value=Brake_Pedal_Pos, sep="_")
    temp2 = subset(temp2, is.na(nTime_60) == F)
    temp3 = subset(temp3, is.na(nTime_60) == F)
    temp=cbind(temp1,temp2,temp3)
    
    fullData = rbind(fullData, temp)
  }
}
write.csv(fullData,"Train_sir.csv")

#balancing the dataset
library(data.table)
library(OSTSC)


train=fread("E:/USA/Projects/Research/Conferance/Individual_subjects/train_wi1.csv")
test=fread("E:/USA/Projects/Research/Conferance/Individual_subjects/test_final.csv")
train=train[,-1]
test=test[,-1]


test_balanced=OSTSC(sample=train[,4:186],label=train[,3])
x_train=as.data.table(test_balanced$sample)
colMeans(x_train[,5])
x_train_scale=scale(x_train)
x_train_scale=as.data.table(x_train_scale)
colMeans(x_train_scale)

y_train=as.data.table(test_balanced$label)
train_balanced_scaled=cbind(y_train,x_train_scale)
write.csv(train_balanced_scaled,"train_balanced_scaled.csv")

#building model




library(h2o)
library(data.table)
library(plyr)
h2o.init(nthreads=-1,enable_assertions = FALSE)

train_balanced_scaled=fread("train_balanced_scaled.csv")
train_balanced_scaled=train_balanced_scaled[,-1]
test_trimmed=fread("test_trimmed.csv")
test_trimmed=test_trimmed[,-1]
test_trimmed_scaled=fread("test_trimmed_scaled.csv")
test_trimmed_scaled=test_trimmed_scaled[,-1]




colnames(train_balanced_scaled)[1]="td"
test_trimmed=test[,3:186]
test_trimmed$td=mapvalues(test_trimmed$td,c('TRUE','FALSE'),to=c('1','0'))
colnames(test_trimmed)=colnames(train_balanced_scaled)

train_balanced_scaled$td=as.factor(train_balanced_scaled$td)

test_trimmed_scaled=as.data.table(scale(test_trimmed[,2:184]))
test_trimmed_scaled=cbind(test_trimmed[,1],test_trimmed_scaled)
test_trimmed_scaled$td=as.factor(test_trimmed_scaled$td)
test_trimmed$td=as.factor(test_trimmed$td)
#write.csv(test_trimmed,"test_trimmed.csv")
#write.csv(test_trimmed_scaled,"test_trimmed_scaled.csv")
#write.csv(train_balanced_scaled,"train_balanced_scaled.csv")

datx = colnames(train_balanced_scaled[,2:184])
daty = colnames(train_balanced_scaled[,1])


trh=as.h2o(train_balanced_scaled,destination_frame = "trh")
tth=as.h2o(test_trimmed,destination_frame="tth")
txh=as.h2o(test_trimmed_scaled,destination_frame = "txh")
#1_ result
model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(150,120,100,80,60,50,20,10,4), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=F,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'txh',
                            l1=0.00158,reproducible = T)



plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)
pr =h2o.predict(model,txh)
df_yhat_test <- as.data.frame(pr)
mlp_abs=cbind(test_trimmed_scaled$td,df_yhat_test)
write.csv(mlp_abs,"mlp_abs.csv")
getwd()


#1
#


#    train     valid 
#0.6787026 0.5958886 


model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "TanhWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.2, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(150,120,100,80,60,50,20,10,4), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = F, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=F,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'txh',
                            l1=0.00158,reproducible = T)



plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)
pr =h2o.predict(model,txh)
df_yhat_test <- as.data.frame(pr)
mlp_abs=cbind(test_trimmed_scaled$td,df_yhat_test)
write.csv(mlp_abs,"mlp_abs.csv")
getwd()

































hyper_params <- list(
  activation=c("RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(150,90,50,10),c(150,120,60,30),c(120,90,60,30),c(150,75,50,25),c(64,64,64,64),c(32,32,32,32),c(64,64,32,16),c(128,64,32,16)),
  input_dropout_ratio=c(0,0.1,0.3,0.2,0.05,0.15,0.25),
  l1=seq(0,1e-2,1e-6),
  l2=seq(0,1e-2,1e-6),distribution = "bernoulli",
  hidden_dropout_ratios=list(c(0.4,0.3,0.3,0.3),c(0.25,0.25,0.25,0.25),c(0.35,0.35,0.35,0.35),c(0.5,0.5,0.5,0.5)))

search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)


model1 <-   h2o.grid(algorithm="deeplearning",x = datx,  # column numbers for predictors
                     y = daty,   # column number for label
                     training_frame = trh,# data in H2O format
                     epochs = 100, # max. no. of epochs
                     standardize = F, 
                     initial_weight_distribution = "Normal" ,
                     stopping_metric="AUC", ## could be "MSE","logloss","r2"
                     nfolds = 5,seed = 1,fold_assignment = 'Modulo',
                     score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 3,
                     score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
                     max_w2=100,                      ## can help improve stability for Rectifier
                     hyper_params = hyper_params,
                     search_criteria = search_criteria,grid_id = "model1",validation_frame = txh)
grid <- h2o.getGrid("model1",sort_by="AUC",decreasing=T)

grid

grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[29]]) ## model with lowest logloss
h2o.saveModel(best_model, path="best_model", force = TRUE)
best_model
plot(best_model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(best_model,train = T,valid = T)
for (i in 1:63){
  best_model <- h2o.getModel(grid@model_ids[[i]])
  j=h2o.auc(best_model,train = T,valid = T)
  print(i)
  print(j)
  
}
best_model <- h2o.getModel(grid@model_ids[[55]])
h2o.saveModel(best_model, path="best_model_0.60_for_abs", force = TRUE)
plot(best_model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(best_model,train = T,valid = T)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")

h2o.auc(model,train = T,valid = T)
pr =h2o.predict(best_model,txh)
df_yhat_test <- as.data.frame(pr)
mlp_abs=cbind(test_trimmed_scaled$td,df_yhat_test)
write.csv(mlp_abs,"mlp_abs3.csv")
getwd()




saved_model=h2o.loadModel("E:\\USA\\Projects\\Research\\R_code\\w2\\best_model_0.774_for_abs\\model1_model_13")

prob=h2o.predict(saved_model,txh)
df=as.data.frame(prob)
roc_auc(df[,3],test_trimmed_scaled[,1])
#0.77777
#
#
#
#
saved_model=h2o.loadModel("E:\\USA\\Projects\\Research\\R_code\\w2\\best_model\\model1_model_37")
prob=h2o.predict(saved_model,txh)
df=as.data.frame(prob)
roc_auc(df[,3],test_trimmed_scaled[,1])
mlp_abs=cbind(test_trimmed_scaled$td,df)
write.csv(mlp_abs,"mlp_abs3.csv")
