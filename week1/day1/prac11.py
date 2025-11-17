from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X, y = make_moons(n_samples=200, noise=0.4, random_state=0)

poly = PolynomialFeatures(degree=10)
X_poly = poly.fit_transform(X)

model = LogisticRegression(C=1, max_iter=5000)

# 5-Fold 교차검증
scores = cross_val_score(model, X_poly, y, cv=5)

print("각 Fold 정확도:", scores)
print("평균 정확도:", np.mean(scores))