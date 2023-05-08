from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

clf = GradientBoostingClassifier(learning_rate=0.1)
clf.fit(X_train, y_train)

print(clf.score(X_test, y_test))
