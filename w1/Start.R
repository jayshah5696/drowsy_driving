setwd("E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train1")
library(data.table)
df1=fread("A_M1001YM01_1S1RNA.csv")
library(sqldf)
demo()

SQLDF <- read.csv.sql("A_M1001YM01_1S1RNA.csv",dbname='dt1', sql = "select * from file WHERE Event_ID
='311'")

?fread
df1=df1[:=,]
head(df1)
df1=df1[df1$Event_ID=='311',]
z=min(df1$win60s)
df2=df1[df1$win60s!=z,]
head(df2)



unique(df1$TA)
df11=df1[df1$TA == '2',]
sample=sample(38377,10000)
sample=seq(1:10000)
df2=df1[sample,]

names(df1)
plot(df11$Steering_Angle~df11$Time)
