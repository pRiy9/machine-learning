#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:05:43 2022

@author: priyanka
"""

# Predicting the house sale prices for King County

## Importing the dataset

import pandas as pd



df = pd.read_csv('/home/priyanka/Downloads/priyanka/Linear Regression/data .csv')

df.head(10)

## Exploring the dataset

### Data type info

df.dtypes

### Dataset Shape

df.shape

### Dataset Columns

df.columns

### Checking missing values

df.isnull().sum()

### Statistical Description of dataset

df.describe()

### Dependent Feature

df['price']

### Importing Visualization library

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

### Distplot

sns.distplot(df['price'])

## Bivariate Analysis

### Correlation

df.corrwith(df['price']).sort_values(ascending = False)

### Heatmap Correlation

fig, ax = plt.subplots(figsize=(20, 20))
sns.heatmap(df.corr(),annot = True, cmap='RdYlGn')

### Visualizing using Scatter plots

fig, ax= plt.subplots(figsize=(27,30), ncols=3, nrows=7)
sns.scatterplot(x="bedrooms", y="price",data=df, ax=ax[0][0])
sns.scatterplot(x="bathrooms", y="price",data=df, ax=ax[0][1])
sns.scatterplot(x="sqft_living", y="price",data=df, ax=ax[0][2])
sns.scatterplot(x="sqft_lot", y="price",data=df, ax=ax[1][0])
sns.scatterplot(x="floors", y="price",data=df, ax=ax[1][1])
sns.scatterplot(x="waterfront", y="price",data=df, ax=ax[1][2])
sns.scatterplot(x="view", y="price",data=df, ax=ax[2][0])
sns.scatterplot(x="condition", y="price",data=df, ax=ax[2][1])
sns.scatterplot(x="grade", y="price",data=df, ax=ax[2][2])
sns.scatterplot(x="sqft_above", y="price",data=df, ax=ax[3][0])
sns.scatterplot(x="sqft_basement", y="price",data=df, ax=ax[3][1])
sns.scatterplot(x="yr_built", y="price",data=df, ax=ax[3][2])
sns.scatterplot(x="yr_renovated", y="price",data=df, ax=ax[4][0])
sns.scatterplot(x="zipcode", y="price",data=df, ax=ax[4][1])
sns.scatterplot(x="lat", y="price",data=df, ax=ax[4][2])
sns.scatterplot(x="long", y="price",data=df, ax=ax[5][0])
sns.scatterplot(x="sqft_living15", y="price",data=df, ax=ax[5][1])
sns.scatterplot(x="sqft_lot15", y="price",data=df, ax=ax[5][2])
sns.scatterplot(x="id", y="price",data=df, ax=ax[6][0])

plt.show()


### Removing irrelevant features

df.drop(['id','date','zipcode','condition','long','sqft_lot15','yr_built','sqft_lot','view','yr_renovated'],axis=1,inplace=True)

df.head()

df.columns

## Splitting the dataset

x = df [['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'grade','sqft_above', 'sqft_basement', 'lat', 'sqft_living15']]
y = df[['price']]

df.head()

df.shape

from sklearn.model_selection import train_test_split 

X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size=0.4)

df.shape

X_train.shape

Y_train.shape

x_test.shape

y_test.shape

df

## Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
x_test = sc.transform(x_test)

## Training & Testing the model

### Linear Regression

from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train, Y_train)

print(lm.intercept_)

predictions = lm.predict(x_test)

# Importing Evaluation metrics

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

import numpy as np
print('R-square score:',r2_score(y_test, predictions))
print('Root Mean squared error:', np.sqrt(mean_squared_error(y_test, predictions)))
print('Mean Absolute error:',mean_absolute_error(y_test, predictions))



