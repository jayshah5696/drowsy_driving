combine =fread("E:/USA/Projects/Research/R_code/w6/pivot_plot.csv",header=T)
colnames(combine)
head(combine)
combine=combine[,-1]
boxplot(combine,main="all data")
