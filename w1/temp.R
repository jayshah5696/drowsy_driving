train=bulk(directory = 'E:/USA/Projects/Research/Conferance/train/60s/train',subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,'train.csv')
rm(train)


t7=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t7", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t7,"t7.csv")
rm(t7)



rm(train1)
m1=min()
attach(train1_311)
uni1=unique(train1_311$RunName)
for (i in uni1){
  
  
}
library(dplyr)
t=train1_311[RunName==i,]
x=min(t$win60s)
t=train1_311[RunName==i & win60s!=x,]
train1=train1_311[train1_311$RunName==uni1[3],]
t3=t2[t2$RunName=='A_M1016YF02_1S2RNA', ]
x=min(t2$win60s)


cran <- getOption("repos")
cran["dmlc"] <- "https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/R/CRAN/GPU"
options(repos = cran)
install.packages("mxnet")


library(mxnet)
a <- mx.nd.ones(c(2,3), ctx = mx.gpu())
b <- a * 2 + 1
b



library(data.table)
library(h2o)
h2o.init(nthreads = -1,max_mem_size='4g')

train=fread("E:/USA/Projects/Research/Conferance/Columns data/All cases/60s/train.csv")
train=train[,-1]
test=fread("E:/USA/Projects/Research/Conferance/Columns data/All cases/60s/test.csv")
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

train_pca=h2o.prcomp(training_frame = 'trh',x=datx,
                     transform = "STANDARDIZE",seed=1001,k=8,max_runtime_secs = 600)



model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                            y = daty,   # column number for label
                            training_frame = "trh",# data in H2O format
                            activation = "MaxoutWithDropout", # or 'Tanh'
                            input_dropout_ratio = 0.0, # % of inputs dropout
                            hidden_dropout_ratios = c(0.5,0.5,0.5), # % for nodes dropout
                            balance_classes = T, 
                            hidden = c(1800,90,10), # three layers of 50 nodes
                            epochs = 100, # max. no. of epochs
                            standardize = T, 
                            initial_weight_distribution = "UniformAdaptive",verbose = F,
                            stopping_metric="AUC", 
                            nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                            score_each_iteration=T,stopping_tolerance = 0.0001,stopping_rounds = 5,
                            validation_frame = 'tth',
                            l1=0.00158)


plot(model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(model,train = T,valid = T)


train_pca=train[,7204:10803]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:55])
pcadata_train=cbind(train[,1:3],prdata$x[,1:55])
write.csv(pcadata_train,'train_prdata_60_steer_label.csv')
prdata_test=predict(prdata,newdata=test[,3604:7203])
prdata_test=cbind(test[,1:3],prdata_test[,1:55])
write.csv(prdata_test,'test_prdata_60_steer_label.csv')

h2o.shutdown(prompt=FALSE)
