#Linear regression in R
library(caTools)
library(ggplot2)

#Import the data
dataset = read.csv('')

#Split the data
split = sample.split(dataset$Y, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fit the model
regressor = lm(formula = X ~ Y, 
               data = training_set)

#Prediction
y_pred = predict(regressor, newdata = test_set)


#Graph the test set and predictions
ggplot() +
  geom_point(aes(x = test_set$X, y = test_set$Y), 
             colour = 'red') +
  geom_line(aes(x = training_set$X, y = predict(regressor, newdata = training_set)),
            colour = 'blue') +
  xlab('') +
  ylab('')