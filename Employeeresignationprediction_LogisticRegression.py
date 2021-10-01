import pandas as pd
import requests
import matplotlib.pyplot as plt
%matplotlib inline
url="https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv"
res=requests.get(url,allow_redirects=True)
with open('insurance_data.csv','wb') as file:
    file.write(res.content)
df=pd.read_csv('insurance_data.csv')

insurance_data.head()

plt.scatter(df.age,df.bought_insurance,marker='+',color='green')

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)

X_test

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)
model.predict(X_test)
model.score(X_test,y_test)
model.predict_proba(X_test)
model.predict([[100]])
