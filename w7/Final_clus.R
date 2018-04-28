### Setting working Directory to Data lOcation
```{r}
setwd("E:/USA/Projects/Research/R_code/w6")
```

```{r}
library(data.table)
library(factoextra)
```
```{r}
train = fread("train_clust.csv",data.table = T)
train = train[,-1]
test = fread("test_clust.csv",data.table = T)
test = test[,-1]
```
Here I have built Custom Function to create Features
```{r}
features = function(data){
    newdata = NULL
    mean_speed = as.data.frame( rep(0,dim(data)[1]))
    mean_acc_lot =as.data.frame( rep(0,dim(data)[1]))
    mean_acc_lan = as.data.frame(rep(0,dim(data)[1]))
    sd_speed = as.data.frame(rep(0,dim(data)[1]))
    sd_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    sd_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    max_speed = as.data.frame(rep(0,dim(data)[1]))
    max_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    max_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    min_speed = as.data.frame(rep(0,dim(data)[1]))
    min_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    min_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    for (i in c(1:dim(data)[1])) {
        mean_speed[i,] = mean(unlist(data[i,4:64]))
        mean_acc_lot[i,] = mean(unlist(data[i , 65:125]))
        mean_acc_lan[i,] = mean(unlist(data[i, 126:186]))
        sd_speed[i,] = sd((unlist(data[ i,4:64])))
        sd_acc_lot[i,] = sd((unlist(data[i , 65:125])))
        sd_acc_lat[i,] = sd((unlist(data[i , 126:186])))
        max_speed[i,] = max((unlist(data[ i,4:64])))
        max_acc_lot[i,] = max((unlist(data[i , 65:125])))
        max_acc_lat[i,] = max((unlist(data[i , 126:186])))
        min_speed[i,] = min((unlist(data[ i,4:64])))
        min_acc_lot[i,] = min((unlist(data[i , 65:125])))
        min_acc_lat[i,] = min((unlist(data[i , 126:186])))
    }
    newdata =as.data.table(cbind(mean_speed,mean_acc_lot,mean_acc_lan, sd_speed,sd_acc_lot,sd_acc_lat,
                                 max_speed,max_acc_lot,max_acc_lat,min_speed,mean_acc_lot,mean_acc_lan))
    colnames(newdata) = c("mean_speed","mean_acc_lot","mean_acc_lan", "sd_speed","sd_acc_lot","sd_acc_lat","max_speed",
                          "max_acc_lot","max_acc_lat","min_speed", "mean_acc_lot","mean_acc_lan")
    return(newdata)
}
```
### Creating Data

```{r}
train_feat = features(train)
test_feat = features(test)
```

```{r}
hc_ward=hclust(dist(train_feat), method="ward.D")

### Questions related to type of Dissimilarity measure to use?

```{r}

plot(hc_ward,main="ward.D", xlab="", sub="", cex=.9)

```
Here we can see only 2 clusters.
```{r}
fviz_nbclust(train_feat, hcut, method = "wss",hc_method = "ward.D", main = "Ward.D") +
  geom_vline(xintercept = 2, linetype = 2)
```



```{r}
index = fread("E:/USA/Projects/Research/R_code/w6/index_all.csv")
index$new = paste(index$RunName,index$win60s)
train$new = paste(train$RunName,train$win60s)

index_train = as.data.table(train$new)
index_train = merge(index_train, index[,c('c','new')],by = 'new')

index_train = cbind(index_train,as.data.table(hc_ward_cut))
```


```{r}
index_train$c = as.factor(index_train$c)
index_train$hc_ward_cut =as.factor(index_train$hc_ward_cut)
write.csv(index_train, "E:/USA/Projects/Research/R_code/w7/index_train.csv")
```

ind = group_by(index_train[,2:3], c) %>% summarize(size = length(hc_ward_cut), frq1 = summary(hc_ward_cut)[1],frq2 = summary(hc_ward_cut)[2])

barplot(height = t(ind[,c(3,4)]), names.arg = ind$c,col=c("blue","red"),legend.text = c("A","B"),args.legend = list(x = "topleft"),axisnames = T,cex.names = 0.5)





event_count = as.data.table(table(index_train$c, dnn = c("events")))
as.data.table(table(index_train$c, dnn = c("events")))
events_more_than_10 = 25 < event_count$N & event_count$N <= 100
events_more_than_10 = event_count$events[events_more_than_10]

events_more_than_100 = event_count$N > 100
events_more_than_100 = event_count$events[events_more_than_100]  



#this is for events having length greater than 100
sample_index = c()
for (i in 1:length(events_more_than_100)){
    set.seed(1001)
    s_index = sample(index_train$new[index_train$c == events_more_than_100[i]],100)
    sample_index = c(sample_index, s_index)
    
}

#this is for variable length less than 100
sampl_ind = c()
for (i in 1:length(events_more_than_10)) {
    s_index = index_train$new[index_train$c == events_more_than_10[i]]
    sampl_ind = c(sampl_ind, s_index)
}

#total sample index for clustering 
total_ind = c(sampl_ind, sample_index)    







temp = as.data.frame(total_ind,col.names = "new")
colnames(temp) = "new"
join_temp = semi_join(index_train, temp, by = "new")
as.data.frame(table(join_temp$c))

mean_speed[i,] = mean(unlist(data[i,4:64]))



shannon.entropy(unlist(test[1, 4:64]))
features = function(data){
    newdata = NULL
    mean_speed = as.data.frame( rep(0,dim(data)[1]))
    mean_acc_lot =as.data.frame( rep(0,dim(data)[1]))
    mean_acc_lan = as.data.frame(rep(0,dim(data)[1]))
    sd_speed = as.data.frame(rep(0,dim(data)[1]))
    sd_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    sd_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    max_speed = as.data.frame(rep(0,dim(data)[1]))
    max_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    max_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    min_speed = as.data.frame(rep(0,dim(data)[1]))
    min_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    min_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    shenen_speed = as.data.frame(rep(0,dim(data)[1]))
    shenen_acc_lot = as.data.frame(rep(0,dim(data)[1]))
    shenen_acc_lat = as.data.frame(rep(0,dim(data)[1]))
    for (i in c(1:dim(data)[1])) {
        mean_speed[i,] = mean(unlist(data[i,4:64]))
        mean_acc_lot[i,] = mean(unlist(data[i , 65:125]))
        mean_acc_lan[i,] = mean(unlist(data[i, 126:186]))
        sd_speed[i,] = sd((unlist(data[ i,4:64])))
        sd_acc_lot[i,] = sd((unlist(data[i , 65:125])))
        sd_acc_lat[i,] = sd((unlist(data[i , 126:186])))
        max_speed[i,] = max((unlist(data[ i,4:64])))
        max_acc_lot[i,] = max((unlist(data[i , 65:125])))
        max_acc_lat[i,] = max((unlist(data[i , 126:186])))
        min_speed[i,] = min((unlist(data[ i,4:64])))
        min_acc_lot[i,] = min((unlist(data[i , 65:125])))
        min_acc_lat[i,] = min((unlist(data[i , 126:186])))
        shenen_speed[i,] = shannon.entropy((unlist(data[ i,4:64])))
        shenen_acc_lot[i,] = shannon.entropy((unlist(data[i , 65:125])))
        shenen_acc_lat[i,] = shannon.entropy((unlist(data[i , 126:186])))
    }
    newdata =as.data.table(cbind(mean_speed,mean_acc_lot,mean_acc_lan, sd_speed,sd_acc_lot,sd_acc_lat,
                                 max_speed,max_acc_lot,max_acc_lat,min_speed,mean_acc_lot,mean_acc_lan,
                                 shenen_speed,shenen_acc_lot,shenen_acc_lat))
    colnames(newdata) = c("mean_speed","mean_acc_lot","mean_acc_lan", "sd_speed","sd_acc_lot","sd_acc_lat","max_speed",
                          "max_acc_lot","max_acc_lat","min_speed", "mean_acc_lot","mean_acc_lan",
                          "shenen_speed","shenen_acc_lot","shenen_acc_lat")
    return(newdata)
}

index_train2 = cbind(events_train[,2],as.data.table(hc_ward_cut2))
index_train2$V1 = as.factor(index_train2$V1)
index_train2$hc_ward_cut2 =as.factor(index_train2$hc_ward_cut2)

table(index_train2$V1)
ind1 = group_by(index_train2, V1) %>% summarize(size = length(hc_ward_cut2), clus1 = summary(hc_ward_cut2)[1],clus2 = summary(hc_ward_cut2)[2])


ind_new1 = ind1 %>% gather(`clus1`, `clus2`, key = cluster, value = count)
ind_new1$size = NULL
ind_new1$V1 = as.factor(ind_new1$V1)
ggplot(ind_new1, aes(x=V1, y=count, fill = cluster, label = "cluster1","cluster2")) + geom_bar(stat="identity",position="stack",width=0.95)+theme_bw() + ylab("Number of Events") +xlab("Events")


write.csv2(ind1, "E:/USA/Projects/Research/R_code/w7/index_2.csv")

plot_colors = c('grey0','grey75','grey35','grey50')
env_fpr_data = ddply(fpr_event_data, .(Environment,Algorithm), function(d){data.frame(Freq=sum(d$Freq))})
env_fpr_data$Environment = factor(env_fpr_data$Environment, levels = c("Rural","Highway","Urban"))

ggplot(index_train2, aes(x=V1, y=hc_ward_cut2)) + geom_bar(stat="identity",position="stack",width=0.95)+coord_flip()+theme_minimal() + ylab("Number of False Positives")




barplot(height = t(ind1[,c(3,4)]), names.arg = ind1$V1,col=c("blue","red"),legend.text = c("1","2"),args.legend = list(x = "topleft"),axisnames = T,cex.names = 0.5)


ggplot(index_train2, aes(x=V1, y=hc_ward_cut2)) + geom_bar(stat = "summary_bin", fun.y = sum,position="stack",width=0.95)+coord_flip()+theme_bw()

+scale_fill_manual(values=plot_colors)



index_train3 = cbind(events_train[,2],as.data.table(hc_ward_cut3))
index_train3$V1 = as.factor(index_train3$V1)
index_train3$hc_ward_cut3 =as.factor(index_train3$hc_ward_cut3)
ind3 = group_by(index_train3, V1) %>% summarize(size = length(hc_ward_cut3), frq1 = summary(hc_ward_cut3)[1],frq2 = summary(hc_ward_cut3)[2],frq3 = summary(hc_ward_cut3)[3])


barplot(height = t(ind3[,c(3,4,5)]), names.arg = ind1$V1,col=c("blue","red","green"),legend.text = c("1","2","3"),args.legend = list(x = "topleft"),axisnames = T,cex.names = 0.5)



index_train4 = cbind(events_train[,2],as.data.table(hc_ward_cut4))
index_train4$V1 = as.factor(index_train4$V1)
index_train4$hc_ward_cut4 =as.factor(index_train4$hc_ward_cut4)

ind3 = group_by(index_train4, V1) %>% summarize(clus1 = sum(hc_ward_cut4==1),clus2 = sum(hc_ward_cut4==2),clus3 = sum(hc_ward_cut4==3), clus4 = sum(hc_ward_cut4==4), clus5 = sum(hc_ward_cut4==5))


ind3 = ind3 %>% gather(`clus1`, `clus2`, `clus3`,`clus4`, key = cluster, value = count)
ind3$size = NULL
ind3$V1 = as.factor(ind3$V1)
ggplot(ind3, aes(x=V1, y=count, fill = cluster, label = "cluster1","cluster2")) + geom_bar(stat="identity",position="stack",width=0.95)+theme_bw() + ylab("Number of Events") +xlab("Events")


#Tasks left
#2) visulising all original time series plot
#3) mean of speed and std deviation of it.
#
#
#
library(cluster)
gap_stat <- clusGap(train_feat, FUN = hcut, nstart = 25, K.max = 10, B = 5, hc_method = "ward.D")


shannon.entropy((unlist(join_train[1 , 65:125])))


library(entropy)
entropy.empirical(unlist(join_train[1 , 65:125]))
, method=c("ML"))

, "MM", "Jeffreys", "Laplace", "SG",
                                  "minimax", "CS", "NSB", "shrink"))
colnames(train_feat)
bar= cbind(train_feat[,1], join_temp[,2])
bar=NULL
bar = cbind(index_train[,2], train[,4:64])
colnames(bar)
61*3
bar$c = as.factor(bar$c)
ggplot(melt(bar,id.vars = "c"), aes(x = c, y = value))+geom_boxplot()

head(bar)
new = melt(bar,id.vars = "c")
summary(new[new$c == 111])
