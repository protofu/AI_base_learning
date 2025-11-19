import matplotlib.pyplot as plt

# 예시 데이터
precision = [1.0, 0.9, 0.8, 0.7, 0.6]
recall = [0.0, 0.2, 0.4, 0.6, 0.8]

plt.plot(recall, precision, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (example)")
plt.grid(True)
plt.show()
