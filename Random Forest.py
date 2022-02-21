#AYŞE BAYRAK
# Random Forest
"""
 Ensemble Yöntemler (Topluluk Öğrenmesi):
     Birden fazla sınıflandıma algoritması veya tahmin algoritması 
aynı anda   kullanılarak daha  başarılı bir  sonuç çıkartabilir. 
Bu kullanıma Ensemble Yöntemler (Topluluk Öğrenmesi) denir .

Ensemble Yöntemler den bir tanesi Random Forestdir.Birden  fazla 
Decision Tree ( Karar ağacı) nın aynı veri kümesi üzerinde çizilmesi
ve hep birlikte kullanılmasına dayanıyor.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler=pd.read_csv("maaslar.csv")

#dataFrame
x=veriler.iloc[:,1:2] 
y=veriler.iloc[:,2:]

#numpy dizi
X=x.values
Y=y.values

from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(random_state=0, n_estimators=10) #n_estimators = kaçtane decision tree çizileceği
rf_reg.fit(X,Y.ravel())

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="green")

print(rf_reg.predict([[6.6]]))