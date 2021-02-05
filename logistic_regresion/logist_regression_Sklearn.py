#Logistic regression with sklearn

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix 

#Importing data
dataset = pd.read_csv('')
X = dataset.iloc[:, [: n-1]].values
y = dataset.iloc[:, n].values

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

#Fit the model
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

#Prediction
y_pred = classifier.predict(X_test)

#Confusion Matrix
cm =confusion_matrix(y_test, y_pred)

