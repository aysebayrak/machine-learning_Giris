#AYŞE BAYRAK
#KARAR AĞACI İLE TAHMİN (DECİSİON TREE)


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


from sklearn.tree import DecisionTreeRegressor
r_dt= DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue")


print(r_dt.predict([[11]]))
print(r_dt.predict([[6.7]]))

    
        
 























