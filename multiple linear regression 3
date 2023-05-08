import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

df = pd.read_csv("input3.csv")

x = df[['Weight', 'Volume']]
y = df['CO2']

xtr, xte, ytr, yte= train_test_split(x, y, test_size= 1/3, random_state=0)

regressor= LinearRegression().fit(xtr, ytr)

yp= regressor.predict(xte)  
xp= regressor.predict(xtr)

plt.plot(xtr, xp, color="red")    
plt.title("Weight, Volume and CO2 (Training Dataset)")  
plt.xlabel("Weight and Volume")  
plt.ylabel("CO2")  
plt.show()
