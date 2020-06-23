#Import packages
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

#Load iris dataset
irisdatasets = datasets.load_iris()
#Split into data and target values
x = irisdatasets.data
y = irisdatasets.target
#Normalize the data
x_norm = (x - x.mean()) / (x.max() - x.min())
y_norm = (y - y.mean()) / (y.max() - y.min())

#Implement Kmeans for k = 3 and k = 50
for j in range(2):
    #First loop, k = 3
    if j == 0:
        k=3
    #Second loop, k = 50
    else:
        k=50
    #Kmeans for clusters 3 for first loop, 50 for second loop
    kmeans = cluster.KMeans(n_clusters=k)
    kmeans.fit(x)
    #Define labels
    labels = kmeans.labels_
    #Define centroids
    centroids = kmeans.cluster_centers_
    #Plot clusters and centroids
    for i in range(k):
        d = x[np.where(labels==i)]
        plt.plot(d[:,0],d[:,1],'o')
        lines = plt.plot(centroids[i,0],centroids[i,1],'kx')
        plt.setp(lines,ms=15.0)
        plt.setp(lines,mew=2.0)
    if j == 0:
        plt.title("KMeans Clusters\n3 Clusters")
    else:
        plt.title("KMeans Clusters\n50 Clusters")
    plt.show()