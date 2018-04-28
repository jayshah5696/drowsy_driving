```{r}
library(data.table)
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
- Should I scale the Data ? ( Mean, std, Min and Max?)

```{r}
hc.complete=hclust(dist(train_feat), method="complete")
hc.average=hclust(dist(train_feat), method="average")
hc.single=hclust(dist(train_feat), method="single")
```
### Questions related to type of Dissimilarity measure to use?
1. Euclidean Distance
2. Correlation based?

```{r}
par(mfrow=c(1,3))
plot(hc.complete,main="Complete Linkage", xlab="", sub="", cex=.9)
plot(hc.average, main="Average Linkage", xlab="", sub="", cex=.9)
plot(hc.single, main="Single Linkage", xlab="", sub="", cex=.9)
```

####  Using New Library

```{r}
plot(hc.complete$height)



library(dendextend)
d_train = dist(train_feat)
d_test = dist(test_feat)
hclust_methods <- c("ward.D", "single", "complete", "average", "mcquitty", 
                    "median", "centroid", "ward.D2")
dd_dendlist <- dendlist()
for(i in seq_along(hclust_methods)) {
  hc_iris <- hclust(d_train, method = hclust_methods[i])   
  dd_dendlist <- dendlist(dd_dendlist, as.dendrogram(hc_iris))
}
names(dd_dendlist) <- hclust_methods
dd_dendlist

dd_dendlist_cor <- cor.dendlist(dd_dendlist)
dd_dendlist_cor


corrplot::corrplot(cor.dendlist(dd_dendlist), "pie", "lower")
```
```{r}
graphics.off()
par(mar=c(1,1,1,1))
par(mfrow = c(4,2))
for(i in 1:8) {
  dd_dendlist[[i]] %>% plot(axes = FALSE, horiz = TRUE)
  title(names(dd_dendlist)[i])
}
%>% set("branches_k_color", k=2) 

```

## Elbow Method
```{r}
library(factoextra)
fviz_nbclust(train_feat, hcut, method = "wss") +
  geom_vline(xintercept = 2, linetype = 2)
```

```{r}
fviz_nbclust(train_feat, hcut, method = "gap_stat",nboot = 15,verbose = T) +
  geom_vline(xintercept = 2, linetype = 2)
```


```{r}
fviz_nbclust(train_feat, hcut, method = "silhouette", hc_method = "average", main = "Average")
fviz_nbclust(train_feat, hcut, method = "silhouette", hc_method = "complete",main = "Complete")
```
```{r}
# Compute gap statistic
set.seed(123)
gap_stat <- clusGap(train_feat, FUN = hcut, K.max = 10, B = 50,hc.method = "complete")
```
```{r}
# Plot gap statistic
fviz_gap_stat(gap_stat)
```
This is for computing all 30 indices using nbclust() package
```{r}
library(NbClust)
nb <- NbClust(train_feat, distance = "euclidean", min.nc = 2, max.nc = 10, method = "complete", index ="all")
```
Visulising the result
```{r}
fviz_nbclust(nb) + theme_minimal()
```



