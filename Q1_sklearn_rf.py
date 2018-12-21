import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

clf = RandomForestClassifier(criterion='entropy')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Entropy) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (# Estimators - 20) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators=40)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (# Estimators - 40) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_depth=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 10) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_depth=8)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 8) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_depth=6)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Depth - 6) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 10) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 20) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=40)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Leaf - 40) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', min_samples_split=40)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 40) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators = 20, min_samples_split=60)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 60) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators =20, min_samples_split=60)
clf = clf.fit(X, y)
acc = clf.score(X_test, y_test)
print('Accuracy (Min Samples Split - 60) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators =20, min_samples_split=60)
clf = clf.fit(X, y)
acc = clf.score(X, y)
print('Accuracy (Min Samples Split - 60) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', min_samples_split=80)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Min Samples Split - 80) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators=20, min_samples_split=60, min_samples_leaf=20)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (# Estimators - 20, Min Samples Split - 60, Min Samples Leaf - 20) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', n_estimators=20, min_samples_split=60, min_samples_leaf=20, max_depth=10)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (# Estimators - 20, Min Samples Split - 60, Min Samples Leaf - 20, Max Depth - 10) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', bootstrap=False)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Bootstrap - False) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features=None)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - None) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features='sqrt')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - sqrt) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features='sqrt', bootstrap=False)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - sqrt, Bootstrap - False) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features='log2')
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - log2) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features=5)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - 5) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features=3)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - 5) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features=7)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - 5) - '+str(acc))

clf = RandomForestClassifier(criterion='entropy', max_features=5, min_samples_split=60)
clf = clf.fit(X, y)
acc = clf.score(X_valid, y_valid)
print('Accuracy (Max Features - 5, Min Samples Split - 60) - '+str(acc))


