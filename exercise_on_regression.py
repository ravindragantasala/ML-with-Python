# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 23:48:19 2018

@author: ravindragantasala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_excel('boston.xls')
X=dataset.drop("MV",axis=1).values
Y=dataset["MV"].values
dataset.isnull().values.any()

#splitting the data set into the train and test data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

#going to build a model on my train data
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)

