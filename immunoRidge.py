import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge
from sklearn import metrics

data =pd.read_csv("Immunotherapy.csv")

print(data.shape)
print(data.isnull().any())
data=data.fillna(method="ffill")

X=data.iloc[:,0:7]
Y=data.iloc[:,-1]

reg=Ridge(alpha=0.1)
reg.fit(X,Y)

print(reg.coef_)