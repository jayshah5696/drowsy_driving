colnames(train_feat)
summary(train_feat$mean_speed)
mean(train_feat$mean_speed)
sd(train_feat$mean_speed)

indices = ifelse(hc_ward_cut==1,TRUE, FALSE)
summary(as.factor(hc_ward_cut))
summary(indices)
clust_1 = train_feat[indices]
clust_2 = train_feat[!indices]
print("Results before Clusters, mean and Std Dev")
mean(train_feat$mean_speed)
sd(train_feat$mean_speed)

print("~~~~~~")

print("mean speed of cluster 1")
mean(clust_1$mean_speed)
print("mean speed of Cluster 2")
mean(clust_2$mean_speed)

print("~~~~~~~~~")

print("Std Dev speed of cluster 1")
sd(clust_1$mean_speed)
print("Std Dev of Cluster 2")
sd(clust_2$mean_speed)

mean(clust_1$sd_speed)
mean(clust_2$sd_speed)

plot(y = fullData[1,4:64], x = rep(1:ncol(fullData[,4:64])),type = 'l')
lines
plot(y = train_feat[1,], x = rep(1:ncol(train_feat)),type  = "l")
ines(y = train_feat[2,], x = rep(1:ncol(train_feat)))

library(ggplot2)
ggplot(fullData, aes(colnames(fullData[4:64])))


input = replicate(
    n = 3,
    matrix(rnorm(n = 240 * 602), nrow = 240, ncol = 602),
    simplify = F
)

plots <- lapply(fullData[4:64], function(mm) { 
    # if you really need to start with data frames, not matrices
    # just put here: mm = as.matrix(mm) 
    df_long = data.frame(id = 1:nrow(mm), index = rep(1:ncol(mm), each = nrow(mm)), value = as.vector(mm))
    ggplot(df_long, aes(x = index, y = value)) +
        geom_line() +
        facet_wrap(~id, nrow = 20) +
        theme(strip.text = element_blank())
})
    

for (i in seq_along(plots)) {
    ggsave(filename = sprintf("myplot%s.png", i),
           plot = plots[[i]],
           height = 30, width = 18)
}
getwd()





t
plot()