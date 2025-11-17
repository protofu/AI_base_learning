import numpy as np
import matplotlib.pyplot as plt

y_true = np.array([1, 2, 3])
y_pred = np.array([1.1, 1.9, 3.2])

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)