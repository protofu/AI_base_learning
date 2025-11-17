from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=200, noise=0.4, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

poly = PolynomialFeatures(degree=10)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 규제 강도 C (작을수록 규제 강함)
for C in [100, 1, 0.1]:
    model = LogisticRegression(C=C, max_iter=5000)
    model.fit(X_train_poly, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_poly))
    test_acc = accuracy_score(y_test, model.predict(X_test_poly))
    print(f"C={C:>5}: Train={train_acc:.3f}, Test={test_acc:.3f}")

