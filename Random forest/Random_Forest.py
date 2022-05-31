#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 31 15:02:09 2022

@author: priyanka
"""

# Student Performance in Exams

### Importing the libraries and dataset

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import pandas as pd

df=pd.read_csv("/home/priyanka/Downloads/priyanka/Random forest/data .csv")

df.head()

### Shape of dataset

print(df.shape)

### Columns and datatype of dataset


list(df.columns)


df.dtypes

### Data information


df.info()

### Statistical Description of Dataset


df.describe()

### Dropping duplicate records if any

df_dup=df.drop_duplicates()

df.shape

### Plotting Heatmap

plt.figure(figsize=(20,10))
plt.title("Heatmap of continuous features",fontweight='bold',fontsize=20)
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',linewidth=1)


print(df['math score'].median(),df['writing score'].median(),df['reading score'].median())

### Adding subject scores to overall_score

df['overall_score']=df['math score']+df['reading score']+df['writing score']
df.head()

### Correlation between Writing score and math score w.r.t gender

sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='gender')

### Correlation between Writing score and math score w.r.t lunch

sns.jointplot(data=df,x='writing score',y='math score',palette='rocket',hue='lunch')

### Correlation between reading score and math score w.r.t gender

sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='gender')

### Correlation between reading score and math score w.r.t lunch

sns.jointplot(data=df,x='reading score',y='math score',palette='rocket',hue='lunch')

### Correlation between reading score and writing score w.r.t gender

sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='gender')

### Correlation between reading score and writing score w.r.t lunch

sns.jointplot(data=df,x='reading score',y='writing score',palette='rocket',hue='lunch')

### Feature Engineering

#### Dropping irrelevant column

df=df.drop(['math score','writing score','reading score'],axis=1)
df.head()

#### Converting object categorical feature into numeric categorical feature.

df['gender']=df['gender'].map({'female':0 , 'male':1}).astype(int)
df['lunch']=df['lunch'].map({'standard':1 , 'free/reduced':0}).astype(int)
df['test preparation course']=df['test preparation course'].map({'none':0 , 'completed':1}).astype(int)
df

### One Hot Encoding

df=pd.get_dummies(df)
df

### Seperating Independent and Dependent features

Y=df['overall_score']
X=df.drop('overall_score',axis=1)
X.head()

### Importing sklearn libraries for building a ML model

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error

#SPLITTING
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#MODEL
model=RandomForestRegressor()
model.fit(x_train,y_train)
    

y_pred = model.predict(x_test)

### Evaluation and Feature Importance

print('RMSE: ', mean_squared_error(y_test, y_pred, squared=False))

feature_importance = np.array(model.feature_importances_)
feature_names = np.array(x_train.columns)
data={'feature_names':feature_names,'feature_importance':feature_importance}
df_plt = pd.DataFrame(data)
df_plt.sort_values(by=['feature_importance'], ascending=False,inplace=True)
plt.figure(figsize=(10,8))
sns.barplot(x=df_plt['feature_importance'], y=df_plt['feature_names'])
plt.xlabel('FEATURE IMPORTANCE')
plt.ylabel('FEATURE NAMES')
plt.show()

