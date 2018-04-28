#extracting all cases of 10s windows. and combining data from multiiple csvs.
setwd("E:/USA/Projects/Research/Conferance")
library(data.table)
library(doParallel)
library(tidyverse)
library(dplyr)
#win60s function highlights
#removing incomplete window data(last)
#removing incomplete window data
#cconverting columns to in row
#and binding multiple csv files from specific folders
bulk=function (directory = ".", subdirectories = FALSE, extension = NULL, 
               data = NULL, verbose = TRUE, fun = utils::read.csv, ...) 
{
  current_setting_warning <- options()$warn
  options(warn = 1)
  if (is.null(data) == FALSE) {
    all_data <- data
  }
  else {
    all_data <- data.table()
  }
  if (class(subdirectories) == "logical") {
    check_subdirectories <- subdirectories
    if (check_subdirectories) {
      subdirectories <- dir(directory)
    }
    else {
      subdirectories <- c("")
    }
  }
  else if (class(subdirectories) == "character") {
    check_subdirectories <- TRUE
  }
  else {
    stop("subdirectories argument should ", "either be boolean or vector of character values")
  }
  for (subdirectory in subdirectories) {
    if (check_subdirectories & verbose) {
      message(paste("Start merging subdirectory:", subdirectory))
    }
    files <- dir(paste(directory, subdirectory, sep = "/"))
    if (is.null(extension) == FALSE) {
      files <- grep(paste0(extension, "$"), files, value = TRUE)
    }
    for (file in files) {
      if (verbose) {
        message(paste("Reading", file))
      }
      single_data <- fun(paste(directory, subdirectory, 
                               file, sep = "/"), ...)
      if (check_subdirectories) {
        single_data$Subdirectory <- subdirectory
      }
      
      single_data$File <- file
      #case310=single_data[single_data=='309']
      #max310=single_data[single_data$win60s]
      #ase310=single_data[single_data$win60s==max310]
      #Exytracting 311 cases and since, first and last windows are incomplete, we are simply removing it(data drain)
     # single_data=single_data[single_data$Event_ID=='311',]
      #single_data=rbind(case310,single_data)
     #
     #
      xmax=max(single_data$win60s)
      single_data=single_data[single_data$win60s!=xmax]
      
      alldata <- data.table()
      single_data[is.na(single_data)] <- 0
      z=unique(single_data$win60s)
      
      
      #converting columns into rows
      for (i in z){
        tr=single_data[single_data$win60s==i, ]
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
      
      all_data <- rbind(all_data, alldata)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}


train=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,"train.csv")
rm(train)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)

#Applying for 30s windows
bulk=function (directory = ".", subdirectories = FALSE, extension = NULL, 
               data = NULL, verbose = TRUE, fun = utils::read.csv, ...) 
{
  current_setting_warning <- options()$warn
  options(warn = 1)
  if (is.null(data) == FALSE) {
    all_data <- data
  }
  else {
    all_data <- data.table()
  }
  if (class(subdirectories) == "logical") {
    check_subdirectories <- subdirectories
    if (check_subdirectories) {
      subdirectories <- dir(directory)
    }
    else {
      subdirectories <- c("")
    }
  }
  else if (class(subdirectories) == "character") {
    check_subdirectories <- TRUE
  }
  else {
    stop("subdirectories argument should ", "either be boolean or vector of character values")
  }
  for (subdirectory in subdirectories) {
    if (check_subdirectories & verbose) {
      message(paste("Start merging subdirectory:", subdirectory))
    }
    files <- dir(paste(directory, subdirectory, sep = "/"))
    if (is.null(extension) == FALSE) {
      files <- grep(paste0(extension, "$"), files, value = TRUE)
    }
    for (file in files) {
      if (verbose) {
        message(paste("Reading", file))
      }
      single_data <- fun(paste(directory, subdirectory, 
                               file, sep = "/"), ...)
      if (check_subdirectories) {
        single_data$Subdirectory <- subdirectory
      }
      
      single_data$File <- file
      #case310=single_data[single_data=='309']
      #max310=single_data[single_data$win60s]
      #ase310=single_data[single_data$win60s==max310]
      #Exytracting 311 cases and since, first and last windows are incomplete, we are simply removing it(data drain)
      # single_data=single_data[single_data$Event_ID=='311',]
      #single_data=rbind(case310,single_data)
      # xmin=min(single_data$win60s)
      #single_data=single_data[single_data$win60s!=xmin]
      xmax=max(single_data$win30s)
      single_data=single_data[single_data$win30s!=xmax]
      
      alldata <- data.table()
      single_data[is.na(single_data)] <- 0
      z=unique(single_data$win30s)
      
      
      #converting columns into rows
      for (i in z){
        tr=single_data[single_data$win30s==i, ]
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
      
      all_data <- rbind(all_data, alldata)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}


train=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,"train.csv")
rm(train)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)



#Applying for 10s windows
bulk=function (directory = ".", subdirectories = FALSE, extension = NULL, 
               data = NULL, verbose = TRUE, fun = utils::read.csv, ...) 
{
  current_setting_warning <- options()$warn
  options(warn = 1)
  if (is.null(data) == FALSE) {
    all_data <- data
  }
  else {
    all_data <- data.table()
  }
  if (class(subdirectories) == "logical") {
    check_subdirectories <- subdirectories
    if (check_subdirectories) {
      subdirectories <- dir(directory)
    }
    else {
      subdirectories <- c("")
    }
  }
  else if (class(subdirectories) == "character") {
    check_subdirectories <- TRUE
  }
  else {
    stop("subdirectories argument should ", "either be boolean or vector of character values")
  }
  for (subdirectory in subdirectories) {
    if (check_subdirectories & verbose) {
      message(paste("Start merging subdirectory:", subdirectory))
    }
    files <- dir(paste(directory, subdirectory, sep = "/"))
    if (is.null(extension) == FALSE) {
      files <- grep(paste0(extension, "$"), files, value = TRUE)
    }
    for (file in files) {
      if (verbose) {
        message(paste("Reading", file))
      }
      single_data <- fun(paste(directory, subdirectory, 
                               file, sep = "/"), ...)
      if (check_subdirectories) {
        single_data$Subdirectory <- subdirectory
      }
      
      single_data$File <- file
      #case310=single_data[single_data=='309']
      #max310=single_data[single_data$win60s]
      #ase310=single_data[single_data$win60s==max310]
      #Exytracting 311 cases and since, first and last windows are incomplete, we are simply removing it(data drain)
      # single_data=single_data[single_data$Event_ID=='311',]
      #single_data=rbind(case310,single_data)
      # xmin=min(single_data$win60s)
      #single_data=single_data[single_data$win60s!=xmin]
      xmax=max(single_data$win10s)
      single_data=single_data[single_data$win10s!=xmax]
      
      alldata <- data.table()
      single_data[is.na(single_data)] <- 0
      z=unique(single_data$win10s)
      
      
      #converting columns into rows
      for (i in z){
        tr=single_data[single_data$win10s==i, ]
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
      
      all_data <- rbind(all_data, alldata)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}


train=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/train", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,"train.csv")
  rm(train)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/Individual_subjects/Individual_subjects/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)


