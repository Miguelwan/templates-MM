#Kmeans clusters

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

#importing data set
dataset = pd.read_csv('')
X = dataset.values

#the elbow method
wcss = []
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')    
plt.ylabel('wcss')
plt.show()

#applying K-means to the dataset
kmeans = KMeans(n_clusters = n, init ='k-means++', max_iter = 300, n_init = 10)
y_kmeans = kmeans.fit_predict(X)

