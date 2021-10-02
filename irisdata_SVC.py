import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
%matplotlib inline

iris=load_iris()
dir(iris)

df=pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()

df['target']=iris.target
iris.target_names
df[df.target==2].head()

df['flower-name']=df.target.apply(lambda x:iris.target_names[x])
df.head()

df0=df[df.target==0]
df1=df[df.target==1]
df2=df[df.target==2]
df1.head()

plt.xlabel('sepal length(cm)')
plt.ylabel('sepal width(cm)')
plt.scatter(df0['sepal length (cm)'],df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'],df1['sepal width (cm)'],color='blue',marker='.')

plt.xlabel('petal length(cm)')
plt.ylabel('petal width(cm)')
plt.scatter(df0['petal length (cm)'],df0['petal width (cm)'],color='green',marker='+')
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='blue',marker='.')

from sklearn.model_selection import train_test_split
X=df.drop(['target','flower-name'],axis='columns')
y=df.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.svm import SVC
model=SVC(gamma=1)
model.fit(X_train,y_train)
model.score(X_test,y_test)
model.predict(X_test)
