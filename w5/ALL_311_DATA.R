library(data.table)
library(tidyverse)
library(dplyr)


#getting dataset
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")

fullData = NULL
drives = list.files()
for(drive in drives){
  temp = fread(drive)[,c("RunName","TD","win60s","Time","Steering_Angle","Event_ID")]
  #temp = temp[temp$Event_ID!=311,]
  #temp = temp[,c(1:5)]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s,Event_ID)
  #temp$Time = round(temp$Time, 1)
  #temp$nTime = round(temp$Time*2) / 2
  temp$nTime = floor(temp$Time*1)
  temp$drowsy = !is.na(temp$TD)
  temp = group_by(temp, RunName,win60s,nTime,Event_ID) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy))
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  #Now need to reshape
  if(max(temp$nTime) <= 60){
    temp = spread(temp, key=nTime,value=Steering_Angle, sep="_")
    temp = subset(temp, is.na(nTime_60) == F)
    fullData = rbind(fullData, temp)
  }
}
#write.csv(fullData[,c(1,2,4)],"E:/USA/Projects/Research/R_code/w6/index_train_all.csv")
#for Index#~~~~~~~~~~~~~~~
full=fullData[,c(1,2,4)]

rm(fullData)

full_temp=group_by(full,RunName, win60s,Event_ID) 
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
full_temp =full %>% mutate(c=Mode(Event_ID))

full_temp= group_by(full_temp,RunName,win60s) %>% summarise(c=median(c))
length(full_temp$c!=311)
unique(full_temp$c)
write.csv(full_temp,"E:/USA/Projects/Research/R_code/w6/index_all.csv")
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``


full=fullData[fullData$Event_ID!=311,]

write.csv(full,"E:/USA/Projects/Research/R_code/w5/train_all.csv")


setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")

fullData = NULL
drives = list.files()
for(drive in drives){
  temp = fread(drive)[,c("RunName","TD","win60s","Time","Steering_Angle","Event_ID")]
  #temp = temp[temp$Event_ID!=311,]
  #temp = temp[,c(1:5)]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s,Event_ID)
  #temp$Time = round(temp$Time, 1)
  #temp$nTime = round(temp$Time*2) / 2
  temp$nTime = floor(temp$Time*1)
  temp$drowsy = !is.na(temp$TD)
  temp = group_by(temp, RunName,win60s,nTime,Event_ID) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy))
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
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
full=fullData[fullData$Event_ID!=311,]

write.csv(fullData,"E:/USA/Projects/Research/R_code/w5/test_all.csv")


###################################################################################################3

  #getting dataset
  setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train")
  
  fullData = NULL
  drives = list.files()
  for(drive in drives){
    temp = fread(drive)[,c("RunName","TD","win60s","Time","Steering_Angle","Event_ID")]
    temp = arrange(temp, Time)
    temp = group_by(temp, RunName, win60s)
    #temp$Time = round(temp$Time, 1)
    #temp$nTime = round(temp$Time*2) / 2
    temp$nTime = floor(temp$Time*1)
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
  
  write.csv(fullData,"E:/USA/Projects/Research/R_code/w5/train_with_all_events.csv")


setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")

fullData = NULL
drives = list.files()
for(drive in drives){
  temp = fread(drive)[,c("RunName","TD","win60s","Time","Steering_Angle","Event_ID")]
  temp = arrange(temp, Time)
  temp = group_by(temp, RunName, win60s)
  #temp$Time = round(temp$Time, 1)
  #temp$nTime = round(temp$Time*2) / 2
  temp$nTime = floor(temp$Time*1)
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
write.csv(fullData,"E:/USA/Projects/Research/R_code/w5/test_with_all_events.csv")
