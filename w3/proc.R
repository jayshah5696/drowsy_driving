library(pROC)
setwd("E:/USA/Projects/Research/R_code/w3")
roc_gasf=read.csv("roc_gasf.csv",header = F)


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
roc_auc(roc_gasf[,4],roc_gasf[,2])


roc_auc_proc=roc(response=roc_gasf[,2],predictor = roc_gasf[,4],auc=T,ci=T,plot = T)
plot.roc(roc_auc_proc)
ci(roc_auc_proc)


roc_gadf=read.csv("roc_gadf.csv",header = F)
roc_auc(roc_gadf[,4],roc_gadf[,2])

roc_auc_proc_gadf=roc(response=roc_gadf[,2],predictor = roc_gadf[,4],auc=T,ci=T,plot = T)
plot.roc(roc_auc_proc_gadf,roc_auc_proc_mtf,roc_auc_proc)
ci(roc_auc_proc_gadf)


roc_mtf=read.csv("roc_mtf.csv",header = F)
roc_auc(roc_mtf[,4],roc_mtf[,2])
roc_auc_proc_mtf=roc(response=roc_mtf[,2],predictor = roc_mtf[,4],auc=T,ci=T,plot = T,smooth = T)
plot.roc(roc_auc_proc_mtf)
ci(roc_auc_proc_mtf)
smooth(roc_auc_proc_mtf)


roc_mlp_abs=read.csv("mlp_abs3.csv",header = T)
roc_mlp_abs=roc_mlp_abs[,-1]
roc_auc(roc_mlp_abs[,4],roc_mlp_abs[,1])
roc_auc_proc_abs=roc(response=roc_mlp_abs[,1],predictor = roc_mlp_abs[,3],auc=T,ci=T,plot = T,smooth = T)
plot.roc(roc_auc_proc_abs)
ci(roc_auc_proc_abs)
smooth(roc_auc_proc_abs,auc=T)



roc_mlp_perclos=read.csv("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/perclos/mlp_perclos.csv",header = T)
roc_mlp_perclos=roc_mlp_perclos[,-1]
roc_auc(roc_mlp_perclos[,4],roc_mlp_perclos[,1])
roc_auc_proc_perclos=roc(response=roc_mlp_perclos[,1],predictor = roc_mlp_perclos[,3],auc=T,ci=T,plot = T,smooth = T)
plot.roc(roc_auc_proc_perclos)
ci(roc_auc_proc_perclos)
smooth(roc_auc_proc_abs,auc=T)




