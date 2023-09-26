import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Immunotherapy.csv")
X=df.iloc[:,0:-1]
Y=df.iloc[:,-1]
corr=X.corr()
fig=plt.figure()
ax=fig.add_subplot(111)
cax=ax.matshow(corr,vmin=-1,vmax=1)
fig.colorbar(cax)

