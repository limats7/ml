import numpy as np
from sklearn.naive_bayes import GaussianNB

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = GaussianNB().fit(X, Y)

print(clf.predict([[-0.8, -1], [10.8, -1]]))

x_test, y_test = [[-0.8, -1], [10.8, -1]], [1, 1]

print("Naive Bayes score:", clf.score(x_test, y_test))
