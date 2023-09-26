import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Immunotherapy.csv")
X1=df.iloc[:,0:4]
X2=df.iloc[:,4:-1]
Y=df.iloc[:,-1]

plt.figure(figsize=(20,20))
Z=X1.plot(kind="box",subplots=True,layout=(2,2),sharex=False,patch_artist=True)
Z=X2.plot(kind="box",subplots=True,layout=(2,2),sharex=False,patch_artist=True)

plt.xticks(rotation=60)
plt.show()