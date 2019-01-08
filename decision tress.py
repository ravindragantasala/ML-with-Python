# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:49:24 2018

@author: ravindragantasala
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Social_Network_Ads.csv")
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

from sklearn.cross_validation import train_test_split
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling datapreprocessing
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X1_train=sc.fit_transform(X1_train)
X1_test=sc.transform(X1_test)

from sklearn.ensemble import RandomForestClassifier
rfrst=RandomForestClassifier(n_estimators =10,criterion="entropy",random_state=0)
rfrst.fit(X1_train,Y1_train)


r_pred=rfrst.predict(Y1_test)

rfrst.score(X1_test,Y1_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y1_test,r_pred)


