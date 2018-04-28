library(data.table)
library(h2o)
library(plyr)
h2o.removeAll()
h2o.init()
train=fread("E:/USA/Projects/Research/Conferance/Columns data/60s/train.csv")
train=train[,-1]
table(train$td)
train_60_steer=train[,7204:10803]
train_60_steer=cbind(train[,1:3],train_60_steer)
rm(train)


test=fread("E:/USA/Projects/Research/Conferance/Columns data/60s/test.csv")
test=test[,-1]
test_60_steer=test[,7204:10803]
test_60_steer=cbind(test[,1:3],test_60_steer)
rm(test)

test_60_steer$td=as.factor(test_60_steer$td)
train_60_steer$td=as.factor(train_60_steer$td)

pca_train=train_60_steer[,4:3603]
prdata=prcomp(pca_train,scale. = T,center = T)

pr.var=prdata$sdev^2

pve=pr.var/sum(pr.var)
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve[1:50]), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
cumsum(pve[1:55])


train=data.table(prdata$x[,1:55])
train=cbind(train_60_steer[,2],train)
train[,1:3]
#applying pca to test data
testpca=predict(prdata,newdata = test_60_steer[,4:3603])
testpca=as.data.frame(testpca[,1:55])
testpca=cbind(test_60_steer[,2],testpca)
testpca[,1:3]


#Balancing data
library(OSTSC)
balanced=OSTSC(train[,2:56],train[,1],parallel=T,progBar = T)
train_balanced=as.data.frame(balanced$sample)
label=as.data.frame(balanced$label)
train_balanced=cbind(label,train_balanced)
colnames(train_balanced)=colnames(train)
train_balanced$td=as.factor(train_balanced$td)
table(train_balanced$td)
train_balanced$td=mapvalues(train_balanced$td,c('1','2'),to=c('0','1'))
table(train_balanced$td)


train_balanced[,1:3]

datx = colnames(train_balanced[,2:56])
daty = colnames(train[,1])
trh=as.h2o(train_balanced,destination_frame = "trh")
tth=as.h2o(testpca,destination_frame="tth")









#training

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(.25,.25,.25,.25,0.25,0.25,0.25,0.25), # % for nodes dropout
                            balance_classes = F, 
                            hidden = c(50,50,50,50,50,50,50,50), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "Normal",verbose = F,
                            stopping_metric="AUC", ## could be "MSE","logloss","r2"
                            nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158,l2=0.008995)


h2o.performance(model)

plot(model, 
     timestep = "epochs", 
     metric = "AUC")
model

#prediction
pr =h2o.predict(model,tth)
df_yhat_test <- as.data.frame(pr)
ups_pred_y = rep("0", length=204) 
ups_pred_y[df_yhat_test[,3] > 0.2] = "1"
table(predict=ups_pred_y,truth=testpca$td)

roc_auc <- function(probabilities,dataset){
  #Command - roc_auc(probabilities,dataset)
  #probabilities are those obtained from predict function
  #dataset is the actual data (0s and 1s)
  library(ROCR)   #Install ROCR library before running 
  pr=prediction(probabilities,dataset)
  prf=performance(pr,measure = "tpr", x.measure = "fpr")
  auc=performance(pr,measure = "auc")
  auc=auc@y.values[[1]]
  plot(prf,colorize=TRUE,main=paste("ROC curve with AUC=",auc))
}

roc_auc(df_yhat_test[,3],testpca[,1])


#hyper
hyper_params <- list(
  activation=c("RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20,20,20),c(50,50,50,50),c(30,30,30,30),c(25,25,25,25),c(64,64,64,64),c(32,32,32,32),c(64,64,32,16),c(64,32,16,8),c(32,32,16,8)),
  l1=seq(0,1e-2,1e-6),
  l2=seq(0,1e-2,1e-6),
  hidden_dropout_ratios=list(c(0.4,0.3,0.3,0.3),c(0.25,0.25,0.25,0.25),c(0.35,0.35,0.35,0.35),c(0.5,0.5,0.5,0.5)))




search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 1800, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)

#training

model1 <-   h2o.grid(algorithm="deeplearning",x = datx,  # column numbers for predictors
                     y = daty,   # column number for label
                     training_frame = trh,# data in H2O format
                     epochs = 100, # max. no. of epochs
                     standardize = T,balance_classes = T, 
                     initial_weight_distribution = "Normal" ,
                     stopping_metric="AUC", ## could be "MSE","logloss","r2"
                     nfolds = 5,seed = 1,fold_assignment = 'Stratified',distribution = "bernoulli",
                     score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 3,
                     score_duty_cycle=0.025,         ## don't score more than 2.5% of the wall time
                     max_w2=10,                      ## can help improve stability for Rectifier
                     hyper_params = hyper_params,
                     search_criteria = search_criteria,grid_id = "model1")
grid <- h2o.getGrid("model1",sort_by="AUC",decreasing=T)
grid


grid@summary_table[1,]
best_model <- h2o.getModel(grid@model_ids[[1]]) ## model with lowest logloss
best_model




pr =h2o.predict(best_model,tth)
df_yhat_test <- as.data.frame(pr)
#ups_pred_y = rep("0", length=204) 
#ups_pred_y[df_yhat_test[,3] > 0.2] = "1"
#table(predict=ups_pred_y,truth=testpca$td)
roc_auc(df_yhat_test[,3],testpca[,1])




#model2

#hyper
hyper_params <- list(
  activation=c("RectifierWithDropout","TanhWithDropout","MaxoutWithDropout"),
  hidden=list(c(20,20,20,20),c(50,50,50,50),c(30,30,30,30),c(25,25,25,25),c(64,64,64,64),c(32,32,32,32),c(64,64,32,16)),
  l1=seq(0,1e-2,1e-6),
  l2=seq(0,1e-2,1e-6),
  hidden_dropout_ratios=list(c(0.4,0.3,0.3,0.3),c(0.25,0.25,0.25,0.25),c(0.35,0.35,0.35,0.35),c(0.5,0.5,0.5,0.5)))




search_criteria = list(strategy = "RandomDiscrete", max_runtime_secs = 360, max_models = 100, seed=1234567, stopping_rounds=5, stopping_tolerance=1e-2)

#training

model1 <-   h2o.grid(algorithm="deeplearning",x = datx,  # column numbers for predictors
                     y = daty,   # column number for label
                     training_frame = trh,# data in H2O format
                     epochs = 100, # max. no. of epochs
                     standardize = T,balance_classes = T, 
                     initial_weight_distribution = "Normal" ,
                     stopping_metric="AUC", ## could be "MSE","logloss","r2"
                     nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                             ## don't score more than 2.5% of the wall time
                     max_w2=10,                      ## can help improve stability for Rectifier
                     hyper_params = hyper_params,
                     search_criteria = search_criteria,grid_id = "model1")
grid <- h2o.getGrid("model1",sort_by="AUC",decreasing=FALSE)
grid


grid@summary_table[6,]
best_model <- h2o.getModel(grid@model_ids[[6]]) ## model with lowest logloss
best_model

h2o.shutdown(prompt=FALSE)
