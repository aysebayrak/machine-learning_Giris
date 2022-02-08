"""
Çoklu Doğrusal Regresyon (Multiple Linear Regression)
   Birden fazla değişkene bağlıdır.
   
Değişken secimi:
    Bütün Değişkenleri Dahil Etme 
    Geriye Doğru Eleme(Backward Elimination)
    İleri Seçim (Forward Selection)
    İki Yönlü Eleme (Bidirection Elimination)
    Skor Karşılaştırması (Score Comparison)
"""
#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kodlar

#2.veri ön işleme

#veri yukleme
veriler=pd.read_csv("veriler.csv")

print(veriler)

Yas=veriler.iloc[:,1:4].values
print(Yas)

#kategorik veriler

#labelEncoder

ulke = veriler.iloc[:,0:1].values
print(ulke)


from sklearn import preprocessing
le= preprocessing.LabelEncoder()
ulke[:,0]=le.fit_transform(veriler.iloc[:,0])
print(ulke)
  
#OneHotEncoder

one=preprocessing.OneHotEncoder()
ulke =one.fit_transform(ulke).toarray()
print(ulke)


#CİNSİYET
c = veriler.iloc[:,-1:].values
print(c)

from sklearn import preprocessing
le= preprocessing.LabelEncoder()
c[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(c)
  
#OneHotEncoder
one=preprocessing.OneHotEncoder()
c =one.fit_transform(c).toarray()
print(c)


#dataFrame
sonuc=pd.DataFrame(data=ulke , index=range(22) , columns=["fr","tr" ,"us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas ,index=range(22) ,columns=["boy","kilo","yas"])
print(sonuc2)


cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data = c[:,:1] ,index=range(22),columns=["cinsiyet"])
print(sonuc3)

#birleştirme

S=pd.concat([sonuc, sonuc2],axis=1)
print(S)

S2=pd.concat([S,sonuc3] ,axis=1)
print(S2)


#eğitim ve test

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(S,sonuc3, test_size=0.33, random_state=0)


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)

y_pred= regressor.predict(x_test)


#Boy kolonunu için predict 
boy=S2.iloc[:,3:4].values
sol=S2.iloc[:,:3]
sag=S2.iloc[:,4:]

veri = pd.concat([sol,sag] ,axis=1)


x_train,x_test, y_train,y_test= train_test_split(veri,boy, test_size=0.33, random_state=0)



regressor2= LinearRegression()
regressor2.fit(x_train,y_train)

y_pred= regressor2.predict(x_test)



#Geriye Doğru Eleme(Backward Elimination)

import statsmodels.api  as sm

X= np.append(arr=np.ones((22,1)).astype(int), values = veri ,axis=1)

X_l= veri.iloc[:,[0,1,2,3,4,5]].values
X_l=np.array(X_l,dtype=float)

model=sm.OLS(boy,X_l).fit()
print(model.summary())



#P value değerine göre en büyük olanı eliyoruz 
X_l= veri.iloc[:,[0,1,2,3,5]].values
X_l=np.array(X_l,dtype=float)

model=sm.OLS(boy,X_l).fit()
print(model.summary())


X_l= veri.iloc[:,[0,1,2,3]].values
X_l=np.array(X_l,dtype=float)

model=sm.OLS(boy,X_l).fit()
print(model.summary())


