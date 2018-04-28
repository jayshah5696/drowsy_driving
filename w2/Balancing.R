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

