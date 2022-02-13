#AYŞE BAYRAK

    # SVR (DESTEK VEKTÖR REGRESYONU)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler=pd.read_csv("maaslar.csv")

#IdataFrame
x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]

#numpy dizi
X=x.values
Y=y.values


# SVR  de veriler ölçeli kullanılması 
from sklearn.preprocessing  import StandardScaler
sc1=StandardScaler()
x_olcekli=sc1.fit_transform(X)
sc2=StandardScaler()
y_olcekli=sc2.fit_transform(Y)


from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli ,y_olcekli)

plt.scatter(x_olcekli,y_olcekli ,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue")

print(svr_reg.predict(11))
print(svr_reg.predict(6))


    
        
 

