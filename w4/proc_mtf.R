library(pROC)
setwd("E:/USA/Projects/Research/R_code/w2")



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




roc_mtf=read.csv("roc_val_0.75_train_0.82.csv",header = T)
roc_mlp_abs=read.csv("E:/USA/Projects/Research/R_code/w3/mlp_abs3.csv",header = T)
roc_mlp_abs=roc_mlp_abs[,-1]
roc_gasf=read.csv("E:/USA/Projects/Research/R_code/w3/roc_gasf.csv",header = F)
roc_gadf=read.csv("E:/USA/Projects/Research/R_code/w3/roc_gadf.csv",header = F)
roc_mlp_perclos=read.csv("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/perclos/mlp_perclos.csv",header = T)
roc_mlp_perclos=roc_mlp_perclos[,-1]



roc_auc_proc_mtf=roc(response=roc_mtf[,2],predictor = roc_mtf[,4],auc=T,ci=T,ci.method = "bootstrap",smooth = T)
roc_auc_proc_perclos=roc(response=roc_mlp_perclos[,1],predictor = roc_mlp_perclos[,3],auc=T,ci=T,ci.method = "bootstrap",smooth = T)
roc_auc_proc_gadf=roc(response=roc_gadf[,2],predictor = roc_gadf[,4],auc=T,ci=T,plot = T,ci.method = "bootstrap",smooth = T)
roc_auc_proc_gasf=roc(response=roc_gasf[,2],predictor = roc_gasf[,4],auc=T,ci=T,plot = T,ci.method = "bootstrap",smooth = T)
roc_auc_proc_abs=roc(response=roc_mlp_abs[,1],predictor = roc_mlp_abs[,3],auc=T,ci=T,plot = T,ci.method = "bootstrap",smooth = T)




##A single plot of all ROC curves
cols = c("#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00")
plot(roc_auc_proc_abs,print.auc=F,main="",xaxt="n",ylab="True Positive Rate", xlab="False Positive Rate",col=cols[1])
plot(roc_auc_proc_gasf,add=T,print.auc=F,main="",xaxt="n",ylab="True Positive Rate", xlab="False Positive Rate",col=cols[2])
plot(roc_auc_proc_gadf,add=T,print.auc=F,main="",xaxt="n",ylab="True Positive Rate", xlab="False Positive Rate",col=cols[3])
plot(roc_auc_proc_mtf,add=T,print.auc=F,main="",xaxt="n",ylab="True Positive Rate", xlab="False Positive Rate",col=cols[5])
plot(roc_auc_proc_perclos,add=T,print.auc=F,main="",xaxt="n",ylab="True Positive Rate", xlab="False Positive Rate",col=cols[4])




legend(0.4,0.6,legend=c(paste("abs: ",round(as.numeric(roc_auc_proc_abs$ci)[2],2),sep=""),
                        paste("perclos: ",round(as.numeric(roc_auc_proc_perclos$ci)[2],2),sep=""),
                        paste("gasf: ",round(as.numeric(roc_auc_proc_gasf$ci)[2],2),sep=""),
                        paste("gadf: ",round(as.numeric(roc_auc_proc_gadf$ci)[2],2),sep=""),
                        paste("mtf: ",round(as.numeric(roc_auc_proc_mtf$ci)[2],2),sep="")),bty="n",col=cols,lwd=2)
axis(1, at=seq(1,0,by=-0.2), labels=c("0.0","0.2","0.4","0.6","0.8","1.0"),pos=-0.04)
