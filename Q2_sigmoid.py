from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt 
import timeit

from visualization import plot_decision_boundary

#-------------GENERAL NEURAL NETWORK IMPLEMENTATION---------------
#Functions for the same

def sigmoid(z):
	return (1/(1+np.exp(-z)))

def grad_sigmoid(z):
	return np.multiply(z, 1-z)

def grad_relu(z):
	res = np.zeros(len(z))
	res[z>0] = 1
	res[z<=0] = 0
	return res

def relu(z):
	res = z
	res[z<=0] = 0
	return res

"""
weights - List of weight matrices
layers - List of outputs at each corresponding layer. Each entry in the form of matrix, with each column correspoding to output for 
particular example in stochastic batch
deltas - List of deltas corresponding to each layer
gradients - List of matrices of gradients of corresponding weights
"""

def cost_function(X, y, weights):
	layers = [0 for i in weights]
	last = len(layers)-1
	for i in range(len(weights)):
		if (i>0):
			# print np.matmul(weights[i], layers[i-1]).shape
			layers[i] = relu(np.matmul(weights[i], layers[i-1]))
		else:
			layers[i] = relu(np.matmul(weights[i], X.T))	
	output = layers[last].T
	output = output.reshape(len(output))
	costF = (np.sum(np.multiply(output-y, output-y)))/(X.shape[0])
	return costF

def accuracy(X, y, weights):
	layers = [0 for i in weights]
	last = len(layers)-1
	for i in range(len(weights)):
		if (i>0):
			layers[i] = relu(np.matmul(weights[i], layers[i-1]))
		else:
			layers[i] = relu(np.matmul(weights[i], X.T))	
	output = layers[last].T
	output = output.reshape(len(output))
	output[output>0.5] = 1
	output[output<=0.5] = 0
	acc = sum(y==output)/len(y)
	return acc

def return_output(X):
	#Forward Propagation
	layers = [0 for i in weights]
	last = len(layers)-1
	for i in range(len(weights)):
		if (i>0):
			layers[i] = sigmoid(np.matmul(weights[i], layers[i-1]))
		else:
			layers[i] = sigmoid(np.matmul(weights[i], X.T))	
	output = layers[last].T
	output = output.reshape(len(output))
	output[output>0.5] = 1
	output[output<=0.5] = 0
	return output

def train_network_single_eg(X, y, X_test, y_test, layers_dim, batch_size, eta=0.1, activation_unit='relu', err = 1e-8):
	training_size = X.shape[0]

	#Weights Creation and Random Initialisation
	weights = []
	for i in range(len(layers_dim)-1):
		weights.append(np.random.randn(layers_dim[i+1], layers_dim[i]))

	layers = [0 for i in weights]
	deltas = [0 for i in weights]
	last = len(layers)-1
	n_iter = 0
	cost_function_arr = []
	train_accuracy_arr = []
	test_accuracy_arr = []
	while True:
		#Stochastic Batch Creation
		batch_idxs = np.random.randint(0, high=training_size, size=batch_size)
		batch = X[batch_idxs, :].T

		for i in range(batch_size):
			x = batch[:, i]
			#Forward Propagation
			for k in range(len(weights)):
				if (activation_unit == 'sigmoid'):
					if (k>0):
						layers[k] = sigmoid(np.matmul(weights[k], layers[k-1]))
					else:
						layers[k] = sigmoid(np.matmul(weights[k], x))
				if (activation_unit == 'relu'):
					if (k == last):
						# print np.matmul(weights[k], layers[k-1])
						layers[k] = sigmoid(np.matmul(weights[k], layers[k-1]))
						# print layers[k]
					elif (k>0):
						# print np.matmul(weights[k], layers[k-1])
						layers[k] = relu(np.matmul(weights[k], layers[k-1]))
						# print layers[k]
					else:
						# print np.matmul(weights[k], x)
						layers[k] = relu(np.matmul(weights[k], x))
						# print layers[k]

			#Backward Propagation
			output = layers[last].T
			y_eg = y[batch_idxs[i]]
			
			#Calculate Deltas for each layer first
			deltas[last] = np.sum(np.multiply(grad_sigmoid(output), (y_eg-output))) #Last Unit will still be sigmoid
			for k in range(last-1, -1, -1):
				dp1 = np.sum(np.multiply(weights[k+1].T, deltas[k+1]), axis=1).T
				dp1.reshape(dp1.size)
				if (activation_unit == 'sigmoid'):	
					dp2 = np.multiply(layers[k], 1-layers[k]) #Uses Sigmoid Units
				if (activation_unit == 'relu'):	
					dp2 = grad_relu(layers[k]) #Uses ReLU Units 
				deltas[k] = np.multiply(dp1, dp2)

			#Calculate weight updates now
			for k in range(last, -1, -1):
				if (k > 0):
					x_d = layers[k-1]
				else:
					x_d = x
				addendum = np.matmul(deltas[k].reshape((deltas[k].size, 1)), x_d.reshape((1, x_d.size)))
				weights[k] = weights[k] + eta*addendum
		n_iter+=1
		eta = 0.1/np.sqrt(n_iter) #Eta directly proportional to underoot of # of iterations

		#Variables for tracking Progress
		print(('Cost Function - '+str(cost_function(X, y, weights))))
		print('Train Accuracy - '+str(accuracy(X, y, weights)))
		print('Valid Accuracy - '+str(accuracy(X_test, y_test, weights)))
		cost_function_arr.append(cost_function(X, y, weights))
		train_accuracy_arr.append(accuracy(X, y, weights))
		test_accuracy_arr.append(accuracy(X_test, y_test, weights))

		#Convergence Criteria
		if n_iter >=2:
			cf1 = cost_function_arr[len(cost_function_arr)-1]
			cf2 = cost_function_arr[len(cost_function_arr)-2]
			if((math.fabs(cf1 - cf2) < err) & (cf1 < cf2)):
				break
			if n_iter >= 4000:
				break
	
	return cost_function_arr, train_accuracy_arr, test_accuracy_arr, weights

#----------------------------------------------------

#--------------------PART(C)-MNIST-------------------

data = np.genfromtxt('mnist_data/MNIST_train.csv', delimiter=',')
print data.shape
X_mnist = data[:, :-1]
X_mnist = X_mnist/255
print X_mnist.shape
y_mnist = data[:, data.shape[1]-1]
print y_mnist
y_mnist = y_mnist.reshape(y_mnist.shape[0])
y_mnist[y_mnist == 6] = 0
y_mnist[y_mnist == 8] = 1

data = np.genfromtxt('mnist_data/MNIST_test.csv', delimiter=',')
print data.shape
X_mnist_test = data[:, :-1]
X_mnist_test = X_mnist_test/255
print X_mnist_test.shape
y_mnist_test = data[:, data.shape[1]-1]
print y_mnist_test
y_mnist_test = y_mnist_test.reshape(y_mnist_test.shape[0])
y_mnist_test[y_mnist_test == 6] = 0
y_mnist_test[y_mnist_test == 8] = 1

layers_dim = [int(X_mnist.shape[1]), 100, 1]

#Sigmoid
start = timeit.default_timer()
cf, tr, te, weights = train_network_single_eg(X_mnist, y_mnist, X_mnist_test, y_mnist_test, layers_dim, 100, activation_unit='sigmoid')
stop = timeit.default_timer()
print(tr)
print(te)
print('Time taken - '+str(stop-start))
plt.figure()
x = np.arange(1, len(tr)+1)
plt.plot(x, tr, 'b-', label='Training Acc')
plt.plot(x, te, 'k-', label='Testing Acc')
plt.xlabel('# of iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('accuracy_mnist_sigmoid.png')

plt.figure()
plt.plot(x, cf, 'b-', label='Cost Function')
plt.xlabel('# of iterations')
plt.ylabel('Cost Function')
plt.legend()
plt.savefig('costf_mnist_sigmoid.png')
