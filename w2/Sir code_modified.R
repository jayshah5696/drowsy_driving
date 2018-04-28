  library(data.table)
  library(tidyverse)
  library(dplyr)
setwd("E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test")

fullData = NULL  #Creating a Data frame
drives = list.files() #getting list of files in working Directory
for(drive in drives){#creating for loop in the wd 
  temp = fread(drive)[,c("Event_ID","RunName","TD","win60s","Time","Steering_Angle","Brake_Pedal_Pos","Acceler_Pedal_Pos")] # reading only 5 Columns from Individual Dataset
  #temp=temp[temp$Event_ID=='311',]
  #xmax=min(temp$win60s)
  #temp=temp[temp$win60s!=xmax]
  temp = arrange(temp, Time)#arranging the dataframe by time column
  temp = group_by(temp, RunName, win60s,Event_ID) #setting up data with grouping time , RunName, win 60s columns
#  temp$nTime = round(temp$Time, 1)
  temp$nTime = round(temp$Time*1) / 2 #creating a reduced frequency time vector with 2HZ
  temp$drowsy = !is.na(temp$TD) #creating a Drowsy Column with True and False
  temp = group_by(temp, RunName,win60s,Event_ID,nTime) %>% summarize(Steering_Angle=median(Steering_Angle, na.rm=T), drowsy=any(drowsy), Brake_Pedal_Pos=median(Brake_Pedal_Pos, na.rm=T), Acceler_Pedal_Pos=median(Acceler_Pedal_Pos, na.rm=T))
  #Reducing Frequency of Data(SteerIng_Angle)
  ##Need to normalize time
  for(i in c(1:(max(temp$win60s)))){
    temp[temp$win60s == i, ]$nTime = temp[temp$win60s == i, ]$nTime - min(temp[temp$win60s ==i, ]$nTime)
  }
  temp = group_by(temp, RunName, win60s,Event_ID)
  #Computes and adds new variable(s). Preserves existing variables. It's similar to the R base function transform().transmute(): Computes new variable(s). Drops existing variables.
  temp = mutate(temp, dd=any(drowsy))
  temp$drowsy = temp$dd
  temp$dd = NULL
  ##Now need to reshape
  if(max(temp$nTime) <= 60){
    
    
    temp1=temp[temp$Event_ID=='311',c("nTime","win60s","RunName","drowsy","Steering_Angle")] 
    xmin=min(temp1$win60s)
    temp1=temp1[temp1$win60s!=xmin,]
    temp1 = spread(temp1, key=nTime,value=Steering_Angle, sep="_")
    temp1 = subset(temp1, is.na(nTime_60) == F)
    temp2=temp[,c("nTime","win60s","RunName","Acceler_Pedal_Pos")] 
    temp2 = spread(temp2, key=nTime,value=Acceler_Pedal_Pos, sep="_")
    temp3=temp[,c("nTime","win60s","RunName","Brake_Pedal_Pos")] 
    temp3 = spread(temp3, key=nTime,value=Brake_Pedal_Pos, sep="_")
    temp2 = subset(temp2, is.na(nTime_60) == F)
    temp3 = subset(temp3, is.na(nTime_60) == F)
    temp=cbind(temp1,temp2,temp3)
    
    fullData = rbind(fullData, temp)
  }
}
write.csv(fullData,"TEst_sir.csv")
