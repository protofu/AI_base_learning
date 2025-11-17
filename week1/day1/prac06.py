import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 공부 시간(X), 합격 여부(y)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])  # 5시간 이후는 합격

# 모델 학습
model = LogisticRegression()
model.fit(X, y)

# 예측 확률
x_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_prob = model.predict_proba(x_test)[:, 1]

plt.plot(x_test, y_prob, label="Success probability (Sigmoid)")
plt.scatter(X, y, color='red', label='real data')
plt.xlabel("study time")
plt.ylabel("Success probability")
plt.legend()
plt.grid(True)
plt.show()

print(model.predict([[4]]))  # 0 (불합격)
print(model.predict([[6]]))  # 1 (합격)
print(model.predict_proba([[6]]))  # [0.2, 0.8] → 80% 확률로 합격
