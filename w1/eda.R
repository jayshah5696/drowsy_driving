train =fread("E:/USA/Projects/Research/Conferance/Columns data/10s/train.csv")
train[,1:5]
train=train[,-1]
train[,1:5]
library(h2o)
h2o.init()
summary(train$td)
train$td=as.factor(train$td)
summary(train$td)



#importing 60s windows
train60 =fread("E:/USA/Projects/Research/Conferance/Columns data/60s/train.csv")
train60$td=as.factor(train60$td)
summary(train60$td)
prdata_60=train60[,-c(1,2,3,4)]
prdata_60[,1:5]
d=as.numeric(prdata_60)
summary(prdata_60[,7200])

#
d=c(1:10800)
d=as.character(d)
colnames(prdata_60)=d
write.csv(prdata_60,'prdata_60.csv')


#acc data pca
prdata_60=train60[,-c(1,2,3,4)]
prdata_60=prdata_60[,1:3600]
pr.out=prcomp(prdata_60)
biplot(pr.out, scale=0)
#scree analysis
pr.out$sdev
pr.var=pr.out$sdev^2
pr.var
pve=pr.var/sum(pr.var)
pve
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve[1:50]), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
cumsum(pve[1:50])

#break data pca
prdata_60=train60[,-c(1,2,3,4)]
prdata_60=prdata_60[,3601:7200]
pr.out=prcomp(prdata_60)
biplot(pr.out, scale=0)
#scree analysis

pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)

plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
cumsum(pve[1:50])

#steering Data
prdata_60=train60[,-c(1,2,3,4)]
prdata_60=prdata_60[,7201:10800]
pr.out=prcomp(prdata_60)
biplot(pr.out, scale=0)
#scree analysis

pr.var=pr.out$sdev^2
pve=pr.var/sum(pr.var)

plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')
plot(cumsum(pve), xlab="Principal Component", ylab="Cumulative Proportion of Variance Explained", ylim=c(0,1),type='b')
cumsum(pve[1:50])




#applying pca
#to 60s Data
train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/60s/train.csv")
table(train$td)

test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/60s/test.csv")
table(test$td)

#acc data pca
train=train[,-1]
test=test[,-1]
#pca for only acc data
train_pca=train[,4:3603]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:13])
pcadata_train=cbind(train[,1:3],prdata$x[,1:13])
write.csv(pcadata_train,'train_prdata_60_acc_label.csv')
prdata_test=predict(prdata,newdata=test[,4:3603])
prdata_test=cbind(test[,1:3],prdata_test[,1:13])
write.csv(prdata_test,'test_prdata_60_acc_label.csv')


#pca for only Break data
train_pca=train[,3604:7203]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:2])
pcadata_train=cbind(train[,1:3],prdata$x[,1:2])
write.csv(pcadata_train,'train_prdata_60_break_label.csv')
prdata_test=predict(prdata,newdata=test[,3604:7203])
prdata_test=cbind(test[,1:3],prdata_test[,1:2])
write.csv(prdata_test,'test_prdata_60_break_label.csv')

#pca for steering Data
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


#to 30s Data


train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/train.csv")
table(train$td)

test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/30s/test.csv")
table(test$td)

#acc data pca
train=train[,-1]
test=test[,-1]
#pca for only acc data
train_pca=train[,4:1803]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:6])
pcadata_train=cbind(train[,1:3],prdata$x[,1:6])
write.csv(pcadata_train,'train_prdata_30_acc_label.csv')
prdata_test=predict(prdata,newdata=test[,4:1803])
prdata_test=cbind(test[,1:3],prdata_test[,1:6])
write.csv(prdata_test,'test_prdata_30_acc_label.csv')


#pca for only Break data
train_pca=train[,1804:3603]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:2])
pcadata_train=cbind(train[,1:3],prdata$x[,1:2])
write.csv(pcadata_train,'train_prdata_30_break_label.csv')
prdata_test=predict(prdata,newdata=test[,1804:3603])
prdata_test=cbind(test[,1:3],prdata_test[,1:2])
write.csv(prdata_test,'test_prdata_30_break_label.csv')

#pca for steering Data
train_pca=train[,3604:5403]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:28])
pcadata_train=cbind(train[,1:3],prdata$x[,1:28])
write.csv(pcadata_train,'train_prdata_30_steer_label.csv')
prdata_test=predict(prdata,newdata=test[,3604:5403])
prdata_test=cbind(test[,1:3],prdata_test[,1:28])
write.csv(prdata_test,'test_prdata_30_steer_label.csv')

#10s
train=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/10s/train.csv")
table(train$td)

test=fread("E:/USA/Projects/Research/Conferance/Columns data/311 cases/10s/test.csv")
table(test$td)

#acc data pca
train=train[,-1]
test=test[,-1]
#pca for only acc data
train_pca=train[,4:603]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:2])
pcadata_train=cbind(train[,1:3],prdata$x[,1:2])
write.csv(pcadata_train,'train_prdata_10_acc_label.csv')
prdata_test=predict(prdata,newdata=test[,4:603])
prdata_test=cbind(test[,1:3],prdata_test[,1:2])
write.csv(prdata_test,'test_prdata_10_acc_label.csv')


#pca for only Break data
train_pca=train[,604:1203]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1])
pcadata_train=cbind(train[,1:3],prdata$x[,1])
write.csv(pcadata_train,'train_prdata_10_break_label.csv')
prdata_test=predict(prdata,newdata=test[,604:1203])
prdata_test=cbind(test[,1:3],prdata_test[,1])
write.csv(prdata_test,'test_prdata_10_break_label.csv')

#pca for steering Data
train_pca=train[,1204:1803]
prdata=prcomp(train_pca,scale=T)
pr.var=prdata$sdev^2
pve=pr.var/sum(pr.var)
cumsum(pve[1:11])
pcadata_train=cbind(train[,1:3],prdata$x[,1:11])
write.csv(pcadata_train,'train_prdata_10_steer_label.csv')
prdata_test=predict(prdata,newdata=test[,1204:1803])
prdata_test=cbind(test[,1:3],prdata_test[,1:11])
write.csv(prdata_test,'test_prdata_10_steer_label.csv')
