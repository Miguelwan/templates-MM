#Linear regresion with pytorch (overkill)
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Data example
n = 100
h = n//2
dimen =2

data = np.random.randn(n, dimen)*3


data[h:,:] = data[h:,:] - 3*np.ones((h,dimen))
data[:h,:] = data[:h,:] + 3*np.ones((h,dimen))

#Tensor mode
target = np.array([0]*h + [1]*h).reshape(n, 1)
x = torch.from_numpy(data).float().requires_grad_(True)
y = torch.from_numpy(target).float()


#Create and train the model
model = nn.Sequential(
        nn.Linear(2,1),
        nn.Sigmoid()
     )

loss_function = nn.BCELoss()

optimizer = optim.SGD(model.parameters(), lr = 0.015)

losses = []
iterations = 2000

for i in range(iterations):
  result = model(x)

  loss = loss_function(result, y)
  losses.append(loss.data)

  optimizer.zero_grad()
  loss.backward()

  optimizer.step()

plt.plot(range(iterations), losses)
loss

#Graph the model
w = list(model.parameters())
w0 = w[0].data.numpy()
w1 = w[1].data.numpy()

plt.scatter(data[:,0], data[:, 1], c=color, s=75, alpha=0.6)

x_axis = np.linspace(-10, 10, n)
y_axis = -(w1[0] + x_axis*w0[0][0])/ w0[0][1]
plt.plot(x_axis, y_axis, 'g--')
