import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import the dataset
dataset=pd.read_excel('boston.xls')
X=dataset.drop('MV',axis=1).values
y=dataset['MV'].values

#Test and Train Sets for regression
from sklearn.cross_validation import train_test_split
X_train,X_test, y_train, y_test= train_test_split(X,y,test_size=0.3,random_state=0)
# Ridge Regression
from sklearn.linear_model import Ridge
ridge=Ridge(alpha=0.1,normalize=True)
ridge.fit(X_train,y_train)
ridge_pred= ridge.predict(X_test)
ridge.score(X_test,y_test)
ridge_coef=ridge.coef_
ridge_intercept=ridge.intercept_

#Linear Regression
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)
regressor_r_value=regressor.score(X_test,y_test)

linear_coef=regressor.coef_

#Lasso Regression
from sklearn.linear_model import Lasso
lasso=Lasso(alpha=0.0007196856730011522)
lasso.fit(X_train,y_train)
lasso_coef=lasso.coef_
lasso_intercept=lasso.intercept_
names=dataset.drop('MV',axis=1).columns 
plt.plot(range(len(names)),lasso_coef)
plt.ylabel('coefficients')
plt.show()
lasso.score(X_test,y_test)

##hyper tunung for lasso
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import train_test_split
lasso.get_params()
c_space=np.logspace(-5,8,15)
param_grid={'alpha':c_space}
logistic_cv=GridSearchCV(lasso,param_grid,cv=5)
logistic_cv.fit(X_train,y_train)
logistic_cv.best_params_
logistic_cv.best_score_





#mean absolute error
from sklearn import metrics
print('MAE:',metrics.mean_absolute_error(y_test,regressor.predict(X_test)))
print('MSE:',metrics.mean_squared_error(y_test,regressor.predict(X_test)))
print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,regressor.predict(X_test))))

#implementing AdjustedR^2
adjusted_r_square=1-(1-regressor_r_value)*(len(y_test)-1/(len(y_test)-X.shape[1]-1))







