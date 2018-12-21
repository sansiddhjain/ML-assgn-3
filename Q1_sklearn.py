import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from read_data import medians, preprocess

filename = "dtree_data/train.csv"
data = preprocess(filename)

X = data[:, 1:]
y = data[:, 0]
y.reshape(len(y))

filename = "dtree_data/valid.csv"
data = preprocess(filename)

X_valid = data[:, 1:]
y_valid = data[:, 0]
y_valid.reshape(len(y_valid))

filename = "dtree_data/test.csv"
data = preprocess(filename)

X_test = data[:, 1:]
y_test = data[:, 0]
y_test.reshape(len(y_test))

clf = DecisionTreeClassifier(criterion='gini')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Gini) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Entropy) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', random_state=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Random State) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_features='sqrt')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - Sqrt) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_features='log2')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - Log2) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=8)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 8) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=6)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 6) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 10) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 20) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=40)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 40) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=60)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 60) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10, Min Samples Leaf - 20) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=8, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 8, Min Samples Leaf - 20) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10, Min Samples Leaf - 10) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=40)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 40) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=60)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 60) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=80)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 80) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=40, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 40, Min Samples Leaf - 20) - '+str(acc))

#BEST FUCKER OUT THERE!!
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=60, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 60, Min Samples Leaf - 20) - '+str(acc))
acc = clf.score(X_test, y_test)
print('Accuracy (Min Samples Split - 60, Min Samples Leaf - 20) - '+str(acc))
acc = clf.score(X, y)
print('Accuracy (Min Samples Split - 60, Min Samples Leaf - 20) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=60, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10, Min Samples Split - 60, Min Samples Leaf - 20) - '+str(acc))

clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_split=40, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10, Min Samples Split - 40, Min Samples Leaf - 20) - '+str(acc))

