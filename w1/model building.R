
library(data.table)
library(h2o)
localH2O = h2o.init(nthreads = -1)
h2o.removeAll()


train=fread("E:/USA/Projects/Research/Conferance/Columns data/60s/train.csv")
#balancing the data
train=train[,-1]
table(train$td)



#60s and steering only balancing

train_60_steer=train[,7204:10803]
train_60_steer=cbind(train[,1:3],train_60_steer)
rm(train)
train_60_steer[,1:5]
library(OSTSC)
balanced=OSTSC(train_60_steer[,4:3603],train_60_steer[ ,2],parallel=T,progBar = T)





test=fread("E:/USA/Projects/Research/Conferance/Columns data/60s/test.csv")
test=test[,-1]
test_60_steer=test[,7204:10803]
test_60_steer=cbind(test[,1:3],test_60_steer)
rm(test)

test_60_steer$td=as.factor(test_60_steer$td)
train_60_steer$td=as.factor(train_60_steer$td)


d=c(1:10800)
d=as.character(d)
colnames(train[,c(4:10803)])=d
colnames(test[,4:10803])=d


datx = colnames(train_60_steer[,4:3603])
daty = colnames(train_60_steer[,2])
trh=as.h2o(train_60_steer,destination_frame = "trh")
tth=as.h2o(test_60_steer,destination_frame="tth")



#training

model <-   h2o.deeplearning(x = datx,  # column numbers for predictors
                   y = daty,   # column number for label
                   training_frame = "trh",# data in H2O format
                   activation = "RectifierWithDropout", # or 'Tanh'
                   input_dropout_ratio = 0.4, # % of inputs dropout
                   hidden_dropout_ratios = c(.5,.5,.5,.5,0.5,0.4,0.4,0.4), # % for nodes dropout
                   balance_classes = F, 
                   hidden = c(2048,1024,512,256,128,64,32,16), # three layers of 50 nodes
                   epochs = 100, # max. no. of epochs
                   standardize = T, 
                   initial_weight_distribution = "Normal",verbose = F,
                   stopping_metric="AUC", ## could be "MSE","logloss","r2"
                   nfolds = 5,seed = 1,fold_assignment = 'Modulo',distribution = "bernoulli",
                   validation_frame = 'tth',score_each_iteration=T,stopping_tolerance = 0.001,stopping_rounds = 3,
                   l1=1e-5)




h2o.shutdown(prompt=FALSE)
