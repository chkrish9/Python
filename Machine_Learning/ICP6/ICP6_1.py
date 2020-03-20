import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from yellowbrick.cluster.elbow import kelbow_visualizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import sklearn

# from sklearn.preprocessing import StandardScaler

#### data preprocessing ######

mydata = pd.read_csv("cc.csv")
ccdata = mydata.dropna()
# y = ccdata.iloc[:, -1]
x = ccdata.iloc[:, 1:]

# print(x)

scaler = StandardScaler()
scaler.fit(x)
x_scaler = scaler.transform(x)

###### checkking with elbow method #########
# kelbow_visualizer(KMeans(random_state=123), x_scaler, k=(2, 10), metric='silhouette')

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, max_iter=300, random_state=0)
    kmeans.fit(x_scaler)
    wcss.append(kmeans.inertia_)
#print(wcss)
plt.plot(range(1, 11), wcss)
plt.title('the elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('Wcss')
plt.show()

### building the kmeans model ###
#
km = KMeans(n_clusters=3, random_state=123)
km.fit(x_scaler)
y_cluster_kmeans = km.predict(x_scaler)
score = metrics.silhouette_score(x_scaler, y_cluster_kmeans)
print(score)
