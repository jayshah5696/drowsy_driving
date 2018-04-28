library(tidyverse)
library(data.table)

setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
drives = list.files()
fulldata = NULL
for (drive in drives){
    temp = fread(drive)[,c("RunName", "Event_ID", "Speed","win60s","Time")]
    temp = arrange(temp, Time)
    temp = group_by(temp, RunName, Event_ID, win60s)
    temp$nTime = floor(temp$Time*1)
    #temp = group_by(temp, RunName, Event_ID, win60s, nTime) %>% summarise(Speed=median(Speed, na.rm = T))
    temp = group_by(temp, RunName, Event_ID, win60s, nTime) %>% summarise(Speed=mean(Speed, na.rm = T))
    fulldata = rbind(fulldata, temp)
}
#temp
#head(fulldata)
#fulldata$win60s = NULL

filter(fulldata, Event_ID %in% c(101,102,103,104,105,106,111,201,202,203,204,205,206,301,302,303,304,305,306,307,308,309,311)) %>% ggplot(aes(x = nTime, y = Speed, group = RunName))+geom_path()+theme_bw()+facet_wrap(~Event_ID)

fulldata$Event_ID = as.factor(fulldata$Event_ID)
filter(fulldata, Event_ID %in% c(101,102,103,104,105,106,111,201,202,203,204,205,206,301,302,303,304,305,306,307,308,309,311)) %>% ggplot(aes(x = Event_ID, y = Speed))+geom_boxplot()+theme_bw()

ggplot(fulldata, aes(x = c, y = value))+geom_boxplot()




temp = filter(train, Event_ID == events_more_than_100[1]) %>%  sample_n(100)

for (col in 5:187){
    join_train[,col] = as.numeric(join_train[,col])
}

join_train[,5:187] <- lapply(join_train[,5:187], function(x) as.numeric(as.character(x)))

min_speed[i,] = min((unlist(data[ i,5:65])))
min_acc_lot[i,] = min((unlist(data[i , 66:126])))
min_acc_lat[i,] = min((unlist(data[i , 127:187])))



library(pracma)
ts =unlist(fullData[1 , 5:65])

entropy_speed = as.data.frame(rep(0,dim(data)[1]))
entropy_acc_lot = as.data.frame(rep(0,dim(data)[1]))
entropy_acc_lat = as.data.frame(rep(0,dim(data)[1]))
approx_entropy(unlist(join_train[3 , 66:126]), edim = 2, r = 0.2*sd(ts), elag = 1)
approx_entropy(unlist(join_train[1 , 127:187]), edim = 2, elag = 1)
entropy_speed = approx_entropy(unlist(data[i , 5:65]), edim = 2, r = 0.2*sd(ts), elag = 1)
entropy_acc_lot = approx_entropy(unlist(data[i , 66:126]), edim = 2, r = 0.2*sd(ts), elag = 1)
entropy_acc_lat = approx_entropy(unlist(data[i , 127:187]), edim = 2, r = 0.2*sd(ts), elag = 1)




library(caret)
preprocessParams <- preProcess(join_train, method=c("scale"))

train_feat1 <- predict(preprocessParams, join_train)
