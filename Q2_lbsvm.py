from __future__ import division
import numpy as np
from svmutil import *

max_val = 255

#Read Training Data
file = np.genfromtxt('mnist_data/MNIST_train.csv', delimiter=',')
X_train = file[:, :-1]
X_train = X_train/max_val
y_train = file[:, -1:]
y_train = y_train.reshape(len(y_train))

X_train = X_train.tolist()
y_train = y_train.tolist()

#Read Testing Data
file = np.genfromtxt('mnist_data/MNIST_test.csv', delimiter=',')
X_test = file[:, :-1]
X_test = X_test/max_val
y_test = file[:, -1:]
y_test = y_test.reshape(len(y_test))

X_test = X_test.tolist()
y_test = y_test.tolist()

# #------------PART (c)------------

#Train using LibSVM

#Linear
m = svm_train(y_train, X_train, '-t 0 -c 1')
svm_save_model('libsvm_partc_linear.model', m)
p_label, p_acc, p_val = svm_predict(y_train, X_train, m)
p_label, p_acc, p_val = svm_predict(y_test, X_test, m)

# #Gaussian
# m = svm_train(y_train, X_train, '-t 2 -g 0.05 -c 1')
# svm_save_model('libsvm_partc_gaussian.model', m)
# p_label, p_acc, p_val = svm_predict(y_test, X_test, m)
