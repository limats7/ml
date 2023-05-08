import numpy as np

x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])
epochs = int(input("Enter the number of epochs: "))
learning_rate = 0.2
w = np.array([float(input("Enter the weight for bias: ")), float(input("Enter the weight for x1: ")), float(input("Enter the weight for x2: "))])

for epoch in range(epochs):
    for i in range(len(x)):
        f = np.dot(x[i], w[1:]) + w[0]
        y_out = 1 if f > 0 else 0
        error = y[i] - y_out
        w[0] += learning_rate * error
        w[1:] += learning_rate * error * x[i]
    print(f"Epoch {epoch+1}: w = {w}")
