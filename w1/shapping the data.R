library(tidyverse)
library(data.table)
train =fread("E:/USA/Projects/Research/DATASET/A_M1001YM01_1S1RNA.csv")



for(i in 1:length(images))
{
  # Read image
  img <- readImage(images[i])
  # Get the image as a matrix
  img_matrix <- img@.Data
  # Coerce to a vector
  img_vector <- as.vector(t(img_matrix))
  # Add label
  vec <- c(label, img_vector)
  # Bind rows
  df <- rbind(df,vec)
  # Print status info
  print(paste("Done ", i, sep = ""))
}

cran <- getOption("repos")
cran["dmlc"] <- "https://s3-us-west-2.amazonaws.com/apache-mxnet/R/CRAN/GPU"
options(repos = cran)
install.packages("mxnet")

sing_data <- data.table()
train[is.na(train)] <- 0
z=unique(train$win60s)
for (i in z){
  tr=train[train$win60s==i, ]
  r=unique(t[,1])
  x=unique(t[,7])
  if (length(x)==1) {
    td=0
  }else {
    td=1
  }
  w=i
  a=t(tr[,3])
  b=t(tr[,4])
  s=t(tr[,5])
  sing=cbind(r,td,w,a,b,s)
  all_data=rbind(all_data,sing)
  
}
colnames(all_data)[1:3]=c("runname","td",'winnum')




names(all_data)
is.na(train)
unique(train$Td)
t=train[train$win60s==i, ]
t[,3]
t[,4]
t[,5]




















single_data=fread("E:/USA/Projects/Research/Conferance/dataset/example/A_M1153OM03_3S1RNA.csv")
#single_data$File <- file
#case310=single_data[single_data=='309']
#max310=single_data[single_data$win60s]
#ase310=single_data[single_data$win60s==max310]
single_data=single_data[single_data$Event_ID=='311',]
#single_data=rbind(case310,single_data)
xmin=min(single_data$win60s)
single_data=single_data[single_data$win60s!=xmin]
xmax=max(single_data$win60s)
single_data=single_data[single_data$win60s!=xmax]

alldata <- data.table()
single_data[is.na(single_data)] <- 0
z=unique(single_data$win60s)

ts[is.na(ts)] <- 0

for (i in z[]){
  tr=ts[ts$win60s==35, ]
  r=unique(tr[,1])
  x=unique(tr[,7])
  if (nrow(x)==1) {
    td=0
  }else {
    td=1
  }
  w=i
  a=t(tr[,3])
  b=t(tr[,4])
  s=t(tr[,5])
  sing=cbind(r,td,w,a,b,s)
  alldata=rbind(alldata,sing)
  
  
}

z[c(5,6)]
alldata[,1:5]





ts=fread("E:/USA/Projects/Research/Conferance/Dataset/example/A_M1153OM03_3S1RNA.csv")
sort(unique(ts$Event_ID))
