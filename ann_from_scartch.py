# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 09:42:12 2018

@author: ravindragantasala
"""

import numpy as np
import matplotlib as plt
import pandas as pd
dataset= pd.read_csv('Churn_Modelling.csv')
x=dataset.iloc[:,3:13].values
y=dataset.iloc[:,13].values


#preprocessing the data

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encode=LabelEncoder()
x[:,1]=encode.fit_transform(x[:,1])
x[:,2]=encode.fit_transform(x[:,2])

ohe=OneHotEncoder(categorical_features=[1])
x=ohe.fit_transform(x).toarray()
x=x[:,1:]

ohe=OneHotEncoder(categorical_features=[2])
x=ohe.fit_transform(x).toarray()

from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)

#doing  the standerd scala ie features scalingr
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)

# now lets make the ANN
#importng the keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
#initializing the ANN
classifier =Sequential()

# adding the input layer and first hidden layer
classifier.add(Dense(output_dim =6,init ='uniform', activation ='relu', input_dim=11))

#adding 2nd hidden layer
classifier.add(Dense(output_dim =6,init ='uniform', activation ='relu'))

# adding the output layer
classifier.add(Dense(output_dim=1,init ='uniform',activation ='sigmoid' ))
# compliling the ANN
classifier.compile(optimizer ='adam',loss='binary_crossentropy',metrics=['accuracy'])

#applying the ANN to the training set
classifier.fit(x_train,y_train,batch_size = 10,nnb_epoch=100)
#predicting the test set results
y_pred=classifier.predict(x_test)
y_pred=(y_pred>0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)











