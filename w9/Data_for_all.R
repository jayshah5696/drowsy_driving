setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
#setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")
fullData = NULL

Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}

drives = list.files()

library(data.table)
library(dplyr)
library(tidyverse)
for(drive in drives) {
    temp = fread(drive)[,c("Event_ID","RunName","TD","win60s","Time","Steering_Angle")]
    temp = arrange(temp, Time)
    temp = group_by(temp, RunName, win60s,Event_ID)
    temp$nTime = floor(temp$Time*1)
    temp$drowsy = !is.na(temp$TD)
    temp = group_by(temp, RunName,win60s,nTime) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy), Event_ID = Mode(Event_ID)) 
    for(i in c(1:(max(temp$win60s)))){
        temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s == i, ]$nTime)
    }
    temp = group_by(temp, RunName, win60s)
    temp = mutate(temp, dd=any(drowsy))
    temp$drowsy = temp$dd
    temp$dd = NULL
    temp = mutate(temp, id = Mode(Event_ID))
    temp$Event_ID = temp$id
    temp$id = NULL
    if(max(temp$nTime) <= 60){
        temp = temp[,c("nTime","win60s","RunName","drowsy","Steering_Angle","Event_ID")]
        temp = spread(temp, key=nTime,value=Steering_Angle, sep="_")
        temp = subset(temp, is.na(nTime_60) == F)
        
        fullData = rbind(fullData, temp)
    }
}


write.csv(fullData,"E:/USA/Projects/Research/R_code/w9/train.csv")
#write.csv(fullData,"E:/USA/Projects/Research/R_code/w9/test_all.csv")
#write.csv(temp1,"E:/USA/Projects/Research/R_code/w6/temp.csv")

