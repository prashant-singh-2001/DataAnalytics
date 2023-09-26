import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import linear_model

data =pd.read_csv("Immunotherapy.csv")
data.shape
print(data.isnull().any())
data=data.fillna(method="ffill")

X=data.iloc[:,0:7]
Y=data.iloc[:,-1]
plt.figure(figsize=(20,20))
plt.tight_layout()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y, train_size=0.2,random_state=0)

reg=linear_model.Lasso(alpha=0.1)
reg.fit(X_train,Y_train)

print(reg.coef_)

Y_pred=reg.predict(X_test)
df=pd.DataFrame({"Actual":Y_test,"Predicted":Y_pred})


print("MSE:",metrics.mean_squared_error(Y_test, Y_pred))


plt.scatter(Y_test, Y_pred,color='gray')
plt.plot(Y_test,Y_pred, color='red',linewidth=1)
plt.show()