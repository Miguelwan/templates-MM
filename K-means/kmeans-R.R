#Kmeans clusters

dataset <- read.csv('')
X <- dataset


#the elbow method
wcss <- vector()
for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type = "b", main = paste('The Elbow Method'), 
     xlab = "Number of Clusters", ylab = "WCSS")


#apply k-mean to the dataset 
kmeans <- kmeans(X, n, iter.max = 300, nstart = 10)

