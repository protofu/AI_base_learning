import numpy as np
import matplotlib.pyplot as plt

# step 1
## 공부시간(x)과 시험점수(y)
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])
plt.scatter(X, y, color='blue')
plt.xlabel("study time (hours)")
plt.ylabel("score")
# plt.show()

# step 2
## 모델 정의(가정 : y = Wx + b)
# 초기값 설정
W = 0.0     # 가중치
b = 0.0     # 절편
lr = 0.01   # 학습률
epochs = 1000

# 경사하강법
for i in range(epochs):
    y_pred = W*X +b
    loss = np.mean((y_pred - y) ** 2)

    dW = np.mean(2 * (y_pred - y) * X)
    db = np.mean(2 * (y_pred - y))

    W -= lr * dW
    b -= lr * db

print(f"W={W:.3f}, b={b:.3f}, 최종 손실={loss:.4f}")

# plt.scatter(X, y, color='blue', label='실제 데이터')
# plt.plot(X, W*X + b, color='red', label='예측 선')
# plt.legend()
# plt.title("Linear Regression")
# plt.show()

from sklearn.linear_model import LinearRegression

model = LinearRegression()
X_reshaped = X.reshape(-1, 1)
model.fit(X_reshaped, y)

print("기울기 W:", model.coef_)
print("절편 b:", model.intercept_)
print("예측:", model.predict([[6]]))