# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:00:12 2018

@author: Dell
"""

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
iris=datasets.load_iris()
type(iris)
print(iris.keys())
type(iris.data)
type(iris.target)
iris.data.shape
iris.target_names
X=iris.data
y=iris.target
df=pd.DataFrame(X,columns=iris.feature_names)
_=pd.scatter_matrix(df,c=y,figsize=[8,8],s=150,marker='D')
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42,stratify=y)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=8)
knn.fit(iris['data'],iris['target'])

y_pred=knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))
#Accuracy
knn.score(X_test,y_test)    
#Confusion Matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#Ploting accuracy
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k) 

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


#KNN Pipelining
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
df=pd.read_csv('white-wine.csv')
#df.info()
X = df.iloc[:, 0:11].values
y = df.iloc[:, 11].values
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
steps=[('scalar',StandardScaler()),('knn',KNeighborsClassifier())]
from sklearn.pipeline import Pipeline
pipeline=Pipeline(steps)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=21)
knn_scaled=pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)
accuracy_score(y_test,y_pred)
knn_unscaled=KNeighborsClassifier().fit(X_train,y_train)  
knn_unscaled.score(X_test,y_test)

#CV and scaling in a pipeline
knn_scaled.get_params()
parameters={'knn__n_neighbors':np.arange(1,50)}
cv=GridSearchCV(pipeline,param_grid=parameters)          
cv.fit(X_train,y_train)
y_pred=cv.predict(X_test)
print(cv.best_params_)
print(cv.score(X_test,y_test))
print(classification_report(y_test,y_pred))







