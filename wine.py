# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 14:37:16 2020

@author: Administrator
"""

#The dataset related to red variants of the Portuguese “Vinho Verde” wine.
#Due to privacy and logistic issues, only physicochemical (inputs) 
#and sensory (the output) variables are available 
#(e.g. there is no data about grape types, wine brand, wine selling price, etc.).
#We will take into account various input features
#like fixed acidity, volatile acidity, citric acid,
#residual sugar, chlorides, free sulfur dioxide, 
#total sulfur dioxide, density, pH, sulphates,
#alcohol. Based on these features we will predict the quality of the wine
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
dataset=pd.read_csv("winequality-white.csv")
dataset.shape
dataset.isnull().any()
dataset = dataset.fillna(method='ffill')

X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = dataset['quality'].values

#plt.figure(figsize=(15,10))
#plt.tight_layout()
#seabornInstance.distplot(dataset['quality'])_
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


#plt.scatter(y_test,y_pred,  color='gray')
#plt.plot(y_test, y_pred, color='red', linewidth=1)
#plt.show()