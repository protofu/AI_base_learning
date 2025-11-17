from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

# 데이터 생성
X, y = make_moons(n_samples=200, noise=0.2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 다항특징으로 복잡도 조절
poly = PolynomialFeatures(degree=10)  # 복잡한 모델
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LogisticRegression(max_iter=5000)
model.fit(X_train_poly, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train_poly))
test_acc = accuracy_score(y_test, model.predict(X_test_poly))

print(f"훈련 정확도: {train_acc:.3f}")
print(f"테스트 정확도: {test_acc:.3f}")
