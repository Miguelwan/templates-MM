#Linear regresion in Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

#Import data
dataset = pd.read_csv('')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#Fit the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Prediction
y_pred = regressor.predict(X_test)

#Graph the test set and prediction
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_train), color = 'blue')
plt.title('')
plt.xlabel('')
plt.ylabel('')
plt.show()
