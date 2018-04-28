

Mode <- function(x) {
    ux <- unique(x)
    ux[which.max(tabulate(match(x, ux)))]
}


setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
#setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")
fullData = NULL

drives = list.files()

library(data.table)
library(dplyr)
library(tidyverse)
for(drive in drives) {
  temp = fread(drive)[,c("RunName","TD","win60s","Time","Speed","Acc_Long","Acc_Lat","Event_ID")]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s,Event_ID)
  temp$nTime = floor(temp$Time*1)
  temp$drowsy = !is.na(temp$TD)
  temp = group_by(temp, RunName,win60s,nTime) %>% summarize(Speed=median(Speed, na.rm = T), Acc_Long=median(Acc_Long, na.rm = T), Acc_Lat = median(Acc_Lat, na.rm = T), drowsy=any(drowsy), Event_ID = Mode(Event_ID)) 
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
    temp1 = temp[,c("nTime","win60s","RunName","drowsy","Speed","Event_ID")]
    temp1 = spread(temp1, key=nTime,value=Speed, sep="_")
    temp1 = subset(temp1, is.na(nTime_60) == F)
    temp2 = temp[,c("nTime","win60s","RunName","Acc_Long")]
    temp2 = spread(temp2, key = nTime, value = Acc_Long, sep = '_')
    temp2 = subset(temp2, is.na(nTime_60) == F)
    temp3 = temp[,c("nTime","win60s","RunName","Acc_Lat")]
    temp3 = spread(temp3, key = nTime, value = Acc_Lat, sep = "_")
    temp3 = subset(temp3, is.na(nTime_60) == F)
    temp = cbind(temp1,temp2,temp3)
    
    fullData = rbind(fullData, temp)
  }
}

colnames(fullData)
fullData=fullData[,c(1:65,68:128,131:191)]
colnames(fullData)
sa=  sprintf("spnTime_%d", 0:60)
colnames(fullData)[5:65]=sa
ap=sprintf("aloTime_%d", 0:60)
colnames(fullData)[66:126]=ap
bp=sprintf("alanTime_%d", 0:60)
colnames(fullData)[127:187]=bp
colnames(fullData)

write.csv(fullData,"E:/USA/Projects/Research/R_code/w8/train_clust.csv")
#write.csv(fullData,"E:/USA/Projects/Research/R_code/w8/test_clust.csv")
#write.csv(temp1,"E:/USA/Projects/Research/R_code/w6/temp.csv")


#for Event Ids only Extra
write.csv(fullData[,"Event_ID"],"E:/USA/Projects/Research/R_code/w7/eventids_train.csv")
