import numpy as np
import matplotlib.pyplot as plt

y_true = np.array([1, 2, 3])
y_pred = np.array([1.1, 1.9, 3.2])

mse = np.mean((y_true - y_pred) ** 2)
print("MSE:", mse)

x = np.linspace(-2, 2, 100)
mse = x**2
mae = np.abs(x)

plt.plot(x, mse, label="MSE (x^2)")
plt.plot(x, mae, label="MAE (|x|)")
plt.legend()
plt.title("MSE vs MAE 손실함수 형태")
plt.show()
