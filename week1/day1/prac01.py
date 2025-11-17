import numpy as np

# 1. 백터 변경
a = np.array([2, 3])
b = np.array([4, 1])

# 2. 내적(Dot product)
dot = np.dot(a, b)
print("내적:", dot) # (2*4 + 3*1) = 11

# 3. 행렬곱
W = np.array([[1, 2], [3, 4]])  # 가중치 행렬
x = np.array([[5], [6]])        # 입력 벡터(열벡터)
y = np.matmul(W, x)
print("행렬곱 결과:\n", y)