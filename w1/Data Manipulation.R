

setwd("E:/USA/Projects/Research/R_code")
# To import all .csv files from the prostate_folder directory:
library(h2o)
h2o.init()
setwd("E:/USA/Projects/Research/Conferance")
trainpath <- system.file("train", package = "h2o")
prosPath <- system.file("extdata", "prostate_folder", package = "h2o")
trainpath="E:/USA/Projects/Research/Conferance/test"
path="https://www.dropbox.com/home/Individual_subjects"
prostate_pattern.hex <- h2o.importFolder(path = trainpath, pattern = ".csv")
class(prostate_pattern.hex)
summary(prostate_pattern.hex)

"/A/.*/iris_.*"
""