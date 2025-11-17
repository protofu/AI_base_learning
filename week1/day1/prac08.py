import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X = np.linspace(0, 1, 10)
y = X + 0.1 * np.random.randn(10)   # 노이즈 포함

plt.scatter(X, y, label='Train Data')

# underfit(직선)
plt.plot(X, X, label='Underfit', color='green')

# good fit(적절한 다항식)
plt.plot(X, X + 0.1*np.sin(5*X), label='Good Fit', color='blue')

# overfit(지나치게 복잡한 곡선)
plt.plot(X, X + 0.2*np.sin(25*X), label='Overfit', color='red')

plt.legend()
plt.title("Underfitting vs Overfitting")
plt.show()