

xtrain =fread("E:/USA/Projects/Research/R_code/w5/weights/train_all.csv")

names(xtrain)
train=xtrain[,5:65]
library(OSTSC)
bal=OSTSC(train, label=xtrain$drowsy)
balanced=as.data.table(bal)
names(balanced)

write.csv(balanced,"E:/USA/Projects/Research/R_code/w5/balanced_all.csv")