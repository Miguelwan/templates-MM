#Linear regresion with pytorch (overkill)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Data example
x = [5, 6, 7, 8, 9, 10]
y = [8.5, 8, 7.5, 7, 6.5, 6]

x_array = np.array(x).reshape(-1,1)
x_array = np.array(y).reshape(-1,1)

#Tensor mode
X = torch.from_numpy(x_array).float().requires_grad_(True)
Y = torch.from_numpy(y_array).float()

#Create and train the model
model = nn.Linear(1, 1)
loss_function = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.015)

losses = []
iterations = 2000

for i in range(iterations):
  pred = model(X)
  loss = loss_function(pred, Y)
  losses.append(loss.data)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

print(loss)
plt.plot(range(iterations), losses)
