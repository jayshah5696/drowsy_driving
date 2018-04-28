library(h20)
h2o.init()

train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/train.csv")
train=train[,-1]
test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/test.csv")
test=test[,-1]
train$td=as.factor(train$td)
test$td=as.factor(test$td)



#selecting only steering data
train1=train[,c(1:3,3604:5403)]
test1=test[,c(1:3,3604:5403)]

#clean colnames
colnames(train1) <- iconv(colnames(train1), to='ASCII', sub='')

datx = colnames(train1[,4:1803])
daty = colnames(train1[,2])
trh=as.h2o(train1,destination_frame = "trh")
tth=as.h2o(test1,destination_frame="tth")


model_nn2 <- h2o.deeplearning(x = datx,
                                                             training_frame = trh,
                                                             model_id = "model_nn",
                                                             autoencoder = TRUE,
                                                             reproducible = F, #slow - turn off for real problems
                                                            ignore_const_cols = FALSE,
                                                             seed = 42,
                                                             hidden = c(1200,600,300,600,1200), 
                                                             epochs = 100,
                                                             activation = "Tanh")




#Convert to autoencoded representation
test_autoenc <- h2o.predict(model_nn, tth)



train_features <- h2o.deepfeatures(model_nn, trh, layer = 2)

pretrained_model=h2o.deeplearning(x=datx,y=daty,
                                  training_frame = 'trh',
                                  validation_frame = 'tth',
                                  hidden=c(1200,600,300,600,1200),
                                  epochs = 20,
                                  stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                  seed=1001,pretrained_autoencoder = 'model_nn',
                                  activation = 'Tanh',
                                  ignore_const_cols=F)


plot(pretrained_model, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model,train = T,valid = T)





model_nn1 <- h2o.deeplearning(x = datx,
                              training_frame = trh,
                              model_id = "model_nn1",
                              autoencoder = TRUE,
                              reproducible = F, #slow - turn off for real problems
                              ignore_const_cols = FALSE,
                              seed = 1001,
                              hidden = c(1200,600,300,100,50), 
                              epochs = 100,
                              activation = "Tanh")
getwd()
h2o.saveModel(model_nn1, path="model_nn1", force = TRUE)



pretrained_model1=h2o.deeplearning(x=datx,y=daty,
                                  training_frame = 'trh',
                                  validation_frame = 'tth',
                                  hidden=c(1200,600,300,100,50),
                                  epochs = 20,
                                  stopping_metric = 'AUC',stopping_rounds = 5,stopping_tolerance = 0.001,
                                  seed=1001,pretrained_autoencoder = 'model_nn1',
                                  activation = 'Tanh',
                                  ignore_const_cols=F)


plot(pretrained_model1, 
     timestep = "epochs", 
     metric = "AUC")
h2o.auc(pretrained_model1,train = T,valid = T)








check.deeplearning_stacked_autoencoder <- function() {  
  # this function builds a vector of autoencoder models, one per layer
  get_stacked_ae_array <- function(training_data,layers,args){  
    vector <- c()
    index = 0
    for(i in 1:length(layers)){    
      index = index + 1
      ae_model <- do.call(h2o.deeplearning, 
                          modifyList(list(x=names(training_data),
                                          training_frame=training_data,
                                          autoencoder=T,
                                          hidden=layers[i]),
                                     args))
      training_data = h2o.deepfeatures(ae_model,training_data,layer=1)
      
      names(training_data) <- gsub("DF", paste0("L",index,sep=""), names(training_data)) 
      vector <- c(vector, ae_model)    
    }
    vector
  }
  
  # this function returns final encoded contents
  apply_stacked_ae_array <- function(data,ae){
    index = 0
    for(i in 1:length(ae)){
      index = index + 1
      data = h2o.deepfeatures(ae[[i]],data,layer=1)
      names(data) <- gsub("DF", paste0("L",index,sep=""), names(data)) 
    }
    data
  }
  
  
  
  
  
  ## Build reference model on full dataset and evaluate it on the test set
  model_ref <- h2o.deeplearning(training_frame=train_hex, x=1:(ncol(train_hex)-1), y=response, hidden=c(10), epochs=1)
  p_ref <- h2o.performance(model_ref, test_hex)
  h2o.logloss(p_ref)
  
  ## Now build a stacked autoencoder model with three stacked layer AE models
  ## First AE model will compress the 717 non-const predictors into 200
  ## Second AE model will compress 200 into 100
  ## Third AE model will compress 100 into 50
  layers <- c(200,100,50)
  args <- list(activation="Tanh", epochs=1, l1=1e-5)
  ae <- get_stacked_ae_array(train, layers, args)
  
  ## Now compress the training/testing data with this 3-stage set of AE models
  train_compressed <- apply_stacked_ae_array(train, ae)
  test_compressed <- apply_stacked_ae_array(test, ae)
  
  ## Build a simple model using these new features (compressed training data) and evaluate it on the compressed test set.
  train_w_resp <- h2o.cbind(train_compressed, train_hex[,response])
  test_w_resp <- h2o.cbind(test_compressed, test_hex[,response])
  model_on_compressed_data <- h2o.deeplearning(training_frame=train_w_resp, x=1:(ncol(train_w_resp)-1), y=ncol(train_w_resp), hidden=c(10), epochs=1)
  p <- h2o.performance(model_on_compressed_data, test_w_resp)
  h2o.logloss(p)
  
  
}

doTest("Deep Learning Stacked Autoencoder", check.deeplearning_stacked_autoencoder)
  