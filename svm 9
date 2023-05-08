import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_csv("Job.csv")
X_train, X_test, y_train, y_test = train_test_split(data.drop('Job Offer', axis=1), data['Job Offer'], test_size = 0.20)

svclassifier = SVC(kernel='linear').fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))
