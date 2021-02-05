#Logistic regression in R
library(caTools)


#Importing the data
dataset = read.csv('')


#Split the data
split = sample.split(dataset$Y, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fit the model
classifier = glm(formula = Y ~ .,
                 family = binomial,
                 data = training_set)

#Prediction
prob_pred = predict(classifier, type = 'response', newdata = test_set[-n]) #n the column of Y
y_pred = ifelse(prob_pred > 0.5, 1, 0)

#Confusion matrix
cm = table(test_set[, n], y_pred)

