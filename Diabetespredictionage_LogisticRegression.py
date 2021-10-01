import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv(r"C:\Users\ajinm\OneDrive\Pictures\Screenshots\diabetes2.csv")
df.head()
plt.scatter(df.Age,df.Outcome,marker='+')

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(df[['Age']],df.Outcome,test_size=0.1)
X_test

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)

model.score(X_test,y_test)
model.predict_proba(X_test)
