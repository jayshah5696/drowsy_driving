setwd("E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train1")
library(data.table)
library(readbulk)
library(doParallel)
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
train1=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train1", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train1_311=train1[train1$Event_ID=='311',]
write.csv(train1,"train1.csv")
rm(train1)
write.csv(train1_311,"train1_311.csv")
rm(train1_311)
gc()


train2=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train2", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train2_311=train2[train2$Event_ID=='311',]
write.csv(train2,"train2.csv")
rm(train2)
write.csv(train2_311,"train2_311.csv")
rm(train2_311)
gc()

train3=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train3", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train3_311=train3[train3$Event_ID=='311',]
write.csv(train3,"train3.csv")
rm(train3)
write.csv(train3_311,"train3_311.csv")
rm(train3_311)
gc()


train4=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train4", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train4_311=train4[train4$Event_ID=='311',]
write.csv(train4,"train4.csv")
rm(train4)
write.csv(train4_311,"train4_311.csv")
rm(train4_311)
gc()


train5=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train5", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train5_311=train5[train5$Event_ID=='311',]
write.csv(train5,"train5.csv")
rm(train5)
write.csv(train5_311,"train5_311.csv")
rm(train5_311)
gc()

train6=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/train6", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
train6_311=train6[train6$Event_ID=='311',]
write.csv(train6,"train6.csv")
rm(train6)
write.csv(train6_311,"train6_311.csv")
rm(train6_311)
gc()



test=bulk(directory = "E:/USA/Projects/Research/DATASET/Individual_subjects/Individual_subjects/test", subdirectories = FALSE, extension = ".csv", data = NULL, verbose = F, fun = data.table::fread)
test_311=test[test$Event_ID=='311',]
write.csv(test,"test.csv")
rm(test)
write.csv(test_311,"test_311.csv")
rm(test_311)
gc()
