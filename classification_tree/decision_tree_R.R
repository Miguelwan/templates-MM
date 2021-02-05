# Decision Tree Classification
library(caTools)
library(rpart)


#Importing the dataset
dataset = read.csv('')


# Splitting the data set into the training ste and test set
split = sample.split(dataset$Y, SplitRatio = 0.80)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Classifier to the Training set
classifier = rpart(formula = Y ~ .,
                   data = training_set)

#Predicting the Test set result
y_pred = predict(classifier, newdata = test_set[-n])


#Making the Confusion Matrix
cm = table(test_set[, n], y_pred)

