import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler=pd.read_csv("odev_tenis.csv")

print(veriler)

#Kategorik Dönüşüm
##tek tek çevirmek yerine tüm kolanları kategorik dönüştürüyor
from sklearn import preprocessing
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

#outlook kısmına oneHotEncoder uyguladık
outlook=veriler2.iloc[:,:1]
from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()
print(outlook)

Outlook=pd.DataFrame(data=outlook ,index=range(14),columns=["ov","ra","sn"])
sonveriler=pd.concat([Outlook,veriler.iloc[:,1:3]],axis=1)
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

#eğitim ve test


from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:], test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)
y_pred= regressor.predict(x_test)



#Geriye Doğru Eleme(Backward Elimination)

sonveriler.iloc[:,1:]

import statsmodels.api  as sm
X= np.append(arr=np.ones((14,1)).astype(int), values = sonveriler.iloc[:,:-1],axis=1)
X_l= sonveriler.iloc[:,[0,1,2,3,4]].values
X_l=np.array(X_l,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary())

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
regressor.fit(x_train,y_train)
y_pred= regressor.predict(x_test)



























