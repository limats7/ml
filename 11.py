from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train a random forest classifier with 5 trees
forest = RandomForestClassifier(n_estimators=5, random_state=1, n_jobs=2)
forest.fit(X_train[:, 2:], y_train)  # Only use the last two features

# Test the accuracy of the classifier
accuracy = forest.score(X_test[:, 2:], y_test)
print(f"Accuracy: {accuracy:.3f}")
