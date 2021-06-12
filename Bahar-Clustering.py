import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering, spectral_clustering

dataset = pd.read_csv('Clustersheet.csv')
# Cluster Data by Age and Spending Score
X = dataset.iloc[:, [2, 4]].values 
# Dendrogram diagram representing a tree to performs hierarchical clustering 
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('"Dendrogram"')
plt.xlabel('"Customers"')
plt.ylabel('"Distances"')
plt.show()

hc = AgglomerativeClustering(n_clusters = 3)
y_hc = hc.fit_predict(X)

# ScatterPlot Clustered Data 
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 50, c = '#ff99ff', label = '1st Cluster')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 50, c = '#ff8533', label = '2nd Cluster')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 50, c = '#80ff80', label = '3th Cluster')
plt.title('"Clustered Data"')
plt.xlabel('"Age"')
plt.ylabel('"Spending Score"')
plt.legend()
plt.show()