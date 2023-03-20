# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 23:13:00 2022

@author: adi_t
"""

# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from math import sqrt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import median
import arima as ar

# load dataset
def parser(x):
	return datetime.strptime(''+x,'%Y-%m-%d')
series = read_csv('C:/Users/adi_t/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.10/used_cars_data.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')

series.plot()
pyplot.show()

autocorrelation_plot(series)
pyplot.show()


# fit model
model = ARIMA(series, order=(0,1,0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(0,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.xlabel("No. of Weeks")
pyplot.ylabel("No of Unsold days")
pyplot.show()

print(metrics.mean_absolute_error(test,predictions))

df = pd.read_csv("C:/Users/adi_t/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Python 3.10/used_cars_data.csv")
#pd.options.display.max_columns = 45
df.head()

df_year = df[(df["year"]>1999) & (df["year"]<2021)].copy()
df_year["year"] = df_year["year"].astype(int)
plt.figure(figsize=(20,5))
plt.xticks(rotation= -35)
sns.countplot(data= df_year,x="year")
plt.show()

df_local = df_year[(df_year['latitude']>24) & (df_year['latitude']<50) & (df_year['longitude']>-125) & (df_year['longitude']<-65)]
plt.figure(figsize=(18,10))
plt.title('Region Compare')
sns.scatterplot(data=df_local, x="longitude",y="latitude", hue='year')
plt.show()


df_year = df[(df["year"]>1999) & (df["year"]<2021)].copy()
df_year["year"] = df_year["year"].astype(int)
plt.figure(figsize=(20,5))
plt.xticks(rotation= -35)
sns.countplot(data= df_year,x="year")
plt.show()