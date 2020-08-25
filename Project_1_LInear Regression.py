# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 21:52:33 2020

@author
"""

#importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt

#Reading the data from your
data = pd.read_csv ("advertising.csv")
data.head()

#To visualise Data
fig, axs = plt.subplots(1, 3, sharey = True)
data.plot(kind = 'scatter', x = 'TV', y ='Sales', ax = axs[0], figsize = (14,7))
data.plot(kind = 'scatter', x = 'Radio', y='Sales', ax = axs[1])
data.plot(kind = 'scatter', x = 'Newspaper', y='Sales', ax = axs[2])

#Creating x and y for Linear Regression
feature_cols = ['TV']
x = data[feature_cols]
y = data.Sales

#Importing Linear Refression Algo
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x, y)

print(lr.intercept_)

print(lr.coef_)

result = 6.974 + 0.005546*50
print(result)

#create a DataFrame with min and max value of the table
X_new = pd.DataFrame({'TV' : [data.TV.min(), data.TV.max()]})
X_new.head()

preds = lr.predict(X_new)
preds

data.plot(kind ='scatter', x = 'TV', y ='Sales')
plt.plot(X_new, preds, c = 'red',linewidth = 3)

import statsmodels.formula.api as smf
lm = smf.ols(formula = 'Sales ~ TV', data = data).fit()
lm.conf_int()

#Finding the probablity values
lm.pvalues

#Finding the R-Squared values
lm.rsquared

#Multilinear Regression
feature_cols = ['TV', 'Radio', 'Newspaper']
X = data[feature_cols]
y = data.Sales

lr = LinearRegression()
lr.fit(X, y)

print(lr.intercept_)
print(lr.coef_)

lm = smf.ols(formula = 'Sales ~ TV + Radio + Newspaper', data=data).fit()

lm.conf_int()
lm.summary()