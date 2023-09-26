# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:40:52 2023

@author: Pandora
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
data=pd.read_csv("Immunotherapy.csv")
X=data.iloc[:,4:7]
Y=data.iloc[:,-1]
plt.figure(figsize=(20,20))
print(data.value_counts())
#fig,ax=plt.subplot(4,4,sharex=True,sharey=True)
X.hist()
plt.show()