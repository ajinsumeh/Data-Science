import pandas as pd
import requests
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/codebasics/py/master/ML/13_kmeans/income.csv'
res = requests.get(url, allow_redirects=True)
with open('income.csv','wb') as file:
    file.write(res.content)
df = pd.read_csv('income.csv')
df.head()

df.rename(columns={'Income($)':'Income'},inplace=True)
plt.scatter(df.Age,df.Income)


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
%matplotlib inline

km=KMeans(n_clusters=3)
y_predicted=km.fit_predict(df[['Age','Income']])
df['cluster']=y_predicted
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

plt.scatter(df1.Age,df1.Income,color='green')
plt.scatter(df2.Age,df2.Income,color='blue')
plt.scatter(df3.Age,df3.Income,color='red')

//There is some problem in this scatter plot because out scaling is not proper. Observe the y axis starting from 40000

scaler=MinMaxScaler()
scaler.fit(df[['Income']])
df['Income']=scaler.transform(df[['Income']])
scaler.fit(df[['Age']])
df['Age']=scaler.transform(df[['Age']])

km=KMeans(n_clusters=3)
y_predict=km.fit_predict(df[['Age','Income']])
y_predict


df['cluster']=y_predict
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]

df1 = df[df.cluster==0]
df2 = df[df.cluster==1]
df3 = df[df.cluster==2]
plt.scatter(df1.Age,df1['Income'],color='green')
plt.scatter(df2.Age,df2['Income'],color='red')
plt.scatter(df3.Age,df3['Income'],color='black')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')


k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(df[['Age','Income']])
    sse.append(km.inertia_)
 sse

plt.scatter(k_rng,sse)


