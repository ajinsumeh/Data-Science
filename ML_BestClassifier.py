
#Using the Loan Sanction Dataset and using different classification aglorithm to find out the best classifier using different evaluation metrics.

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline



!wget -O loan_train.csv https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/loan_train.csv



df = pd.read_csv('loan_train.csv')
df.head()


#Convert to date time object

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()

df['loan_status'].value_counts()

!conda install -c anaconda seaborn -y


import seaborn as sns

bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()



bins = np.linspace(df.age.min(), df.age.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'age', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()



df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()


#One Hot Encoding
df.groupby(['education'])['loan_status'].value_counts(normalize=True)
df[['Principal','terms','age','Gender','education']].head()



Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()


X = Feature
X[0:5]
y = df['loan_status'].values
y[0:5]

#Normalizing Data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


#K Means Algorithm
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)
from sklearn.neighbors import KNeighborsClassifier
k=7
neigh=KNeighborsClassifier(n_neighbors=k).fit(X_train,y_train)
yhat=neigh.predict(X_test)
yhat[0:5]
from sklearn import metrics
print("Train Set Accuracy:",metrics.accuracy_score(y_train,neigh.predict(X_train)))
print("Test Set Accuracy:",metrics.accuracy_score(y_test,yhat))



ks=10
mean_acc=np.zeros((ks-1))
std_acc=np.zeros((ks-1))
for  n in range(1,ks):
    neigh=KNeighborsClassifier(n_neighbors=n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1]=metrics.accuracy_score(y_test,yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc
plt.plot(range(1,ks),mean_acc,'g')
plt.fill_between(range(1,ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel("Accuracy")
plt.xlabel("No of neighbors(k)")
plt.tight_layout()
plt.show()
print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
import matplotlib.image as mpimg
from sklearn import tree
from sklearn.model_selection import train_test_split
LoanTree=DecisionTreeClassifier(criterion='entropy',max_depth=4)
LoanTree.fit(X_train,y_train)
predTree=LoanTree.predict(X_test)
print(y_test[0:5])



from sklearn import metrics
print("Decision Tree Accuracy",metrics.accuracy_score(y_test,predTree))

!pip install graphviz
!pip install pydotplus
import graphviz 
import pydotplus
#Visualising the DecisionTree
dot_data = StringIO()
filename = "tree.png"
dot_data = StringIO()
filename = "tree.png"
featureNames = Feature.columns
out=tree.export_graphviz(LoanTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

#Support Vector Machines
from sklearn.svm import SVC
m=SVC(gamma=1)
m.fit(X_train,y_train)
m.score(X_test,y_test)
y=m.predict(X_test)
from sklearn.metrics import f1_score
f1_score(y_test,y, average="weighted")

#Logistic Regression
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
a=model.predict(X_test)
model.predict_proba(X_test[0:5])


#Model Evaluation using Test Set
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
!wget -O loan_test.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_test.csv
  
#Data Pre-processing
test_df = pd.read_csv('loan_test.csv')
test_df.head()
test_df['due_date'] = pd.to_datetime(df['due_date'])
test_df['effective_date'] = pd.to_datetime(df['effective_date'])
test_df.head()

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)
test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
Feature1 = test_df[['Principal','terms','age','Gender','weekend']]
Feature1 = pd.concat([Feature1,pd.get_dummies(test_df['education'])], axis=1)
Feature1.drop(['Master or Above'], axis = 1,inplace=True)
test_X = preprocessing.StandardScaler().fit(Feature1).transform(Feature1)
test_X[0:5]
ytest = test_df['loan_status'].values
ytest[0:5]
test_X1 = preprocessing.StandardScaler().fit(Feature1).transform(Feature1)
test_X1[0:5]
test_Y1=test_df['loan_status'].values
test_Y1[0:5]

#Model Prediction and Testing for best classifier

test_predict1=neigh.predict(test_X1)
jc1=jaccard_score(test_Y1, test_predict1,pos_label = "PAIDOFF")
fs1=f1_score(test_Y1, test_predict1, average='weighted')

test_predict2=LoanTree.predict(test_X1)
jc2=jaccard_score(test_Y1, test_predict2,pos_label = "PAIDOFF")
fs2=f1_score(test_Y1, test_predict2, average='weighted')

test_predict3=m.predict(test_X1)
jc3=jaccard_score(test_Y1, test_predict3,pos_label = "PAIDOFF")
fs3=f1_score(test_Y1, test_predict3, average='weighted')

test_predict4=model.predict(test_X1)
probability=model.predict_proba(test_X1)
jc4=jaccard_score(test_Y1, test_predict4,pos_label = "PAIDOFF")
fs4=f1_score(test_Y1, test_predict4, average='weighted')
logloss=log_loss(test_Y1,probability)

list_jc = [jc1, jc2, jc3, jc4]
list_fs = [fs1, fs2, fs3, fs4]
list_ll = ['NA', 'NA', 'NA', logloss]

df = pd.DataFrame(list_jc, index=['KNN','Decision Tree','SVM','Logistic Regression'])
df.columns = ['Jaccard']
df.insert(loc=1, column='F1-score', value=list_fs)
df.insert(loc=2, column='LogLoss', value=list_ll)
df.columns.name = 'Algorithm'
df
#OUTPUT
# Algorithm 	Jaccard 	F1-score 	LogLoss
# KNN 	0.686275 	0.673636 	NA
# Decision Tree 	0.395349 	0.538580 	NA
# SVM 	0.666667 	0.592593 	NA
# Logistic Regression 	0.740741 	0.630418 	0.548248

































