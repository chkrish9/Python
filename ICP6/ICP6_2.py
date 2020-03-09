import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
#from yellowbrick.cluster.elbow import kelbow_visualizer

#### data preprocessing ######

mydata = pd.read_csv('cc.csv')
ccdata = mydata.dropna()
y = ccdata.iloc[:, -1]
x = ccdata.iloc[:, 1:-1]
# print(x)
### building the PCA model ###

scaler = StandardScaler()

scaler.fit(x)

x_scaler = scaler.transform(x)
pca = PCA(2)
x_pca = pca.fit_transform(x_scaler)

df2 = pd.DataFrame(data=x_pca)
finaldf = pd.concat([df2, ccdata[['TENURE']]], axis=1)
#print(finaldf)

############## Applying kmeans after PCA #################
#
pcadata = finaldf.dropna()
y = pcadata.iloc[:, -1]
x = pcadata.iloc[:, 0:-1]
#
# ###### checkking with elbow method #########
# kelbow_visualizer(KMeans(random_state=123), x, k=(2, 7), metric='silhouette')
#
km = KMeans(n_clusters=3, random_state=123)
km.fit(x)

y_cluster_kmeans = km.predict(x)

score = metrics.silhouette_score(x, y_cluster_kmeans)
print(score)
