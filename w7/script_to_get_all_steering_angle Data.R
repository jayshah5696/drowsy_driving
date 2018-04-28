library(data.table)
library(tidyverse)
#setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")

fullData = NULL
drives = list.files()
for(drive in drives){
    temp = fread(drive)[,c("RunName","TD","win60s","Time","Steering_Angle")]
    temp = arrange(temp, Time)
    temp = group_by(temp, RunName, win60s)
    temp$nTime = floor(temp$Time)
    #temp$nTime = round(temp$Time*2) / 2
    temp$drowsy = !is.na(temp$TD)
    temp = group_by(temp, RunName,win60s,nTime) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy))
    ##Need to normalize time
    for(i in c(1:(max(temp$win60s)))){
        temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
    }
    temp = group_by(temp, RunName, win60s)
    temp = mutate(temp, dd=any(drowsy))
    temp$drowsy = temp$dd
    temp$dd = NULL
    ##Now need to reshape
    if(max(temp$nTime) <= 60){
        temp = spread(temp, key=nTime,value=Steering_Angle, sep="_")
        temp = subset(temp, is.na(nTime_60) == F)
        fullData = rbind(fullData, temp)
    }
}
#write.csv(fullData,"E:/USA/Projects/Research/R_code/train_all_steer.csv")
write.csv(fullData,"E:/USA/Projects/Research/R_code/test_all_steer.csv")
