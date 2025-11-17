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

from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'max_iter': [2000, 5000]
}

grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_poly, y)

print("최적 하이퍼파라미터:", grid.best_params_)
print("평균 정확도:", grid.best_score_)