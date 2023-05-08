import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("input3.csv")

d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)
d = {'yes': 1, 'no': 0}
df['Buy_Computer'] = df['Buy_Computer'].map(d)

features = ['Age', 'Experience', 'Rank', 'Nationality']

X = df[features]
y = df['Buy_Computer']
#dtree = DecisionTreeClassifier()
dtree = DecisionTreeClassifier(criterion="entropy",max_depth=3)

dtree = dtree.fit(X, y)
cn=['No','Yes']
tree.plot_tree(dtree, feature_names=features,class_names=cn)

res=dtree.predict([[18, 9, 7, 1]])
if(res==0):
  print("No")
else:
  print("Yes")

res=dtree.predict([[30, 14, 9, 0]])
if(res==0):
  print("No")
else:
  print("Yes")
