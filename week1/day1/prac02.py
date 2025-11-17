import numpy as np
import matplotlib.pyplot as plt

x = 5.0         # 초기값
lr = 0.1        # 학습률(Learning rate)
history = []

for i in range(20):
    grad = 2 * x            # y = x² → dy/dx = 2x
    x = x - lr * grad       # 경사하강법
    history.append(x)

plt.plot(history, marker='o')
plt.title("Gradient Descent (y = x²)")
plt.xlabel("Iteration")
plt.ylabel("x value")
plt.show()

print("최종 x :", x)