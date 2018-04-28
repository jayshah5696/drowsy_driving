setwd("E:/USA/Projects/Research/Conferance/train")
library(data.table)
library(readbulk)
library(doParallel)
#win60s
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
      single_data=single_data[single_data$Event_ID=='311',]
      x=min(single_data$win60s)
      single_data=single_data[single_data$win60s!=x]
      all_data <- dplyr::bind_rows(all_data, single_data)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}
t1=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t1", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t1,"t1.csv")
rm(t1)

t2=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t2", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t2,"t2.csv")
#A_M1016YF02_1S2RNA
rm(t2)



t3=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t3", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t3,"t3.csv")
rm(t3)


t4=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t4", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t4,"t4.csv")
rm(t4)


t5=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t5", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t5,"t5.csv")
rm(t5)


t6=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t6", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t6,"t6.csv")
rm(t6)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)

train=bulk(directory = 'E:/USA/Projects/Research/Conferance/train/60s/train',subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,'train.csv')
rm(train)



#win10s trimmed
setwd("E:/USA/Projects/Research/Conferance/train")

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
      single_data=single_data[single_data$Event_ID=='311',]
      x=min(single_data$win10s)
      single_data=single_data[single_data$win10s!=x]
      all_data <- dplyr::bind_rows(all_data, single_data)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}

t1=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t1", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t1,"t1.csv")
rm(t1)

t2=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t2", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t2,"t2.csv")
#A_M1016YF02_1S2RNA
rm(t2)



t3=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t3", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t3,"t3.csv")
rm(t3)


t4=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t4", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t4,"t4.csv")
rm(t4)


t5=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t5", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t5,"t5.csv")
rm(t5)


t6=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t6", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t6,"t6.csv")
rm(t6)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)

train=bulk(directory = 'E:/USA/Projects/Research/Conferance/train/30s/train',subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,'train.csv')
rm(train)



#win30s
#win30s trimmed
setwd("E:/USA/Projects/Research/Conferance/train")

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
      single_data=single_data[single_data$Event_ID=='311',]
      x=min(single_data$win30s)
      single_data=single_data[single_data$win30s!=x]
      all_data <- dplyr::bind_rows(all_data, single_data)
    }
  }
  options(warn = current_setting_warning)
  results <- data.table(all_data)
  if (nrow(results) == 0) {
    warning("Final data.frame has 0 rows. ", "Please check that directory was specified correctly.")
  }
  return(results)
}

t1=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t1", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t1,"t1.csv")
rm(t1)

t2=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t2", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t2,"t2.csv")
#A_M1016YF02_1S2RNA
rm(t2)



t3=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t3", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t3,"t3.csv")
rm(t3)


t4=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t4", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t4,"t4.csv")
rm(t4)


t5=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t5", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t5,"t5.csv")
rm(t5)


t6=bulk(directory = "E:/USA/Projects/Research/Conferance/train/t6", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(t6,"t6.csv")
rm(t6)


test=bulk(directory = "E:/USA/Projects/Research/Conferance/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(test,"test.csv")
rm(test)

train=bulk(directory = 'E:/USA/Projects/Research/Conferance/train/30s/train',subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
write.csv(train,'train.csv')
rm(train)




