import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

data = pd.read_csv('Mall_Customers.csv', usecols=[3, 4])
dendrogram = sch.dendrogram(sch.linkage(data, method='single'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
