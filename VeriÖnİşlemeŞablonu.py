#1.kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#kodlar

#2.veri ön işleme

#veri yukleme
veriler=pd.read_csv("eksikveriler.csv")

print(veriler)

#eksik veriler
from sklearn.impute import SimpleImputer  #ekisk veri ort ile hesaplama
imputer=SimpleImputer(missing_values=np.nan, strategy="mean")


print("yaşşşşşşş")
     
Yas=veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4]) # eğitmek için 
Yas[:,1:4]= imputer.transform(Yas[:,1:4]) # nan ile uygula 
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

#dataFrame
sonuc=pd.DataFrame(data=ulke , index=range(22) , columns=["fr","tr" ,"us"])
print(sonuc)

sonuc2=pd.DataFrame(data=Yas ,index=range(22) ,columns=["boy","kilo","yas"])
print(sonuc2)

cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3=pd.DataFrame(data=cinsiyet ,index=range(22),columns=["cinsiyet"])
print(sonuc3)

#birleştirme

S=pd.concat([sonuc, sonuc2],axis=1)
print(S)
S2=pd.concat([S,sonuc3] ,axis=1)
print(S2)


#eğitim ve test

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test= train_test_split(S,sonuc3, test_size=0.33, random_state=0)

#öznitelik ölçeklendirme
    #birbirinden farklı aralıktaki değerlei kullanabilemek için  
    #aynı aralıklara getirmemiz lazım 
    
from sklearn.preprocessing  import StandardScaler
sc=StandardScaler()

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)



