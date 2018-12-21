from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt 

from visualization import plot_decision_boundary
from sklearn.linear_model import LogisticRegression

def sigmoid(z):
	return (1/(1+np.exp(-z)))

def grad_sigmoid(z):
	return np.multiply(z, 1-z)

def grad_relu(z):
	res = np.zeros(len(z))
	res[z>0] = 1
	res[z<=0] = 0
	return res

f = open('input_parta.txt', 'r')
lines = f.readlines()

input_size = int(lines[0][:-1])
hidden_layers_dim = lines[1][1:-2].split(', ')
hidden_layers_dim = list(map(int, hidden_layers_dim))
batch_size = int(lines[2][:-1])

layers_dim = [input_size]+hidden_layers_dim[:]+[1] #The one added since the output has exactly 1 element

"""
weights - List of weight matrices
layers - List of outputs at each corresponding layer. Each entry in the form of matrix, with each column correspoding to output for 
particular example in stochastic batch
deltas - List of deltas corresponding to each layer
gradients - List of matrices of gradients of corresponding weights
"""

def forward_prop(X, y, weights):
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
	# print(output)
	# print(y)
	costF = (np.sum(np.multiply(output-y, output-y)))/(X.shape[0])
	return costF

def accuracy(X, y, weights):
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
	# print output
	# print y
	# print sum(y==output)
	acc = sum(y==output)/len(y)
	return acc

print(layers_dim)
def train_network(X, y, layers_dim, batch_size, eta=0.1):
	training_size = X.shape[0]

	#Weights Creation and Random Initialisation
	weights = []
	for i in range(len(layers_dim)-1):
		weights.append(np.random.randn(layers_dim[i+1], layers_dim[i]))

	layers = [0 for i in weights]
	deltas = [0 for i in weights]
	last = len(layers)-1
	n_iter = 0
	accuracy_arr = []
	while ((accuracy_arr[len(accuracy_arr)-1]>=accuracy_arr[len(accuracy_arr)-2]) & (n_iter>1)):
		#Stochastic Batch Creation
		batch_idxs = np.random.randint(0, high=training_size, size=batch_size)
		batch = X[batch_idxs, :].T

		# print(weights)
		#Forward Propagation
		for i in range(len(weights)):
			if (i>0):
				layers[i] = sigmoid(np.matmul(weights[i], layers[i-1]))
				# print((layers[i]))
			else:
				layers[i] = sigmoid(np.matmul(weights[i], batch))
				# print((layers[i]))

		#Backward Propagation
		output = layers[last].T
		# print(output)
		y_batch = y[batch_idxs]
		# print(y_batch)

		#Calculate Deltas for each layer first
		# deltas[last] = np.sum(np.multiply(grad_sigmoid(output), (y_batch-output)))/batch_size
		deltas[last] = np.sum(np.multiply(grad_sigmoid(output), (y_batch/output)-(1-y_batch/1-output)))/batch_size
		for k in range(last-1, -1, -1):
			dp1 = np.sum(np.multiply(weights[k+1].T, deltas[k+1]).T, axis=1).T
			dp2 = np.sum(np.multiply(layers[k], 1-layers[k]), axis=1) #Uses Sigmoid Units (CHANGE FOR RELU)
			deltas[k] = np.multiply(dp1, dp2)/batch_size

		#Calculate weight updates now
		for k in range(last, -1, -1):
			if (k > 0):
				x = np.sum(layers[k-1], axis=1)/batch_size
			else:
				x = np.sum(batch, axis=1)/batch_size
			# print(k)
			# print(x)
			# print((deltas[k]))
			# print((deltas[k].T))
			if k != last:
				# addendum = np.matmul(x.reshape((len(x), 1)), deltas[k].reshape((1, len(deltas[k]))))
				addendum = np.matmul(deltas[k].reshape((len(deltas[k]), 1)), x.reshape((1, len(x))))
			else:
				addendum = x*deltas[k].T
			weights[k] = weights[k] + eta*addendum
		n_iter+=1

		print('Cost Function - '+str(forward_prop(X, y, weights)))
		print('Accuracy - '+str(accuracy(X, y, weights)))
		accuracy_arr.append(accuracy(X, y, weights))

def train_network_single_eg(X, y, X_test, y_test, layers_dim, batch_size, eta=0.1, err=1e-8):
	training_size = X.shape[0]

	#Weights Creation and Random Initialisation
	weights = []
	for i in range(len(layers_dim)-1):
		weights.append(np.random.randn(layers_dim[i+1], layers_dim[i]))

	layers = [0 for i in weights]
	deltas = [0 for i in weights]
	last = len(layers)-1
	n_iter = 0
	train_accuracy_arr = []
	test_accuracy_arr = []
	cost_function_arr = []
	while n_iter<=600:
		#Stochastic Batch Creation
		batch_idxs = np.random.randint(0, high=training_size, size=batch_size)
		batch = X[batch_idxs, :].T

		# print(weights)
		for i in range(batch_size):
			x = batch[:, i]
			# print(x)
			#Forward Propagation
			for k in range(len(weights)):
				if (k>0):
					layers[k] = sigmoid(np.matmul(weights[k], layers[k-1]))
					# print((layers[k]))
				else:
					layers[k] = sigmoid(np.matmul(weights[k], x))
					# print((layers[k]))

			#Backward Propagation
			output = layers[last].T
			# print(output)
			y_eg = y[batch_idxs[i]]
			# print(y_eg)

			#Calculate Deltas for each layer first
			deltas[last] = np.sum(np.multiply(grad_sigmoid(output), (y_eg-output)))
			# deltas[last] = np.sum(np.multiply(grad_sigmoid(output), (y_batch/output)-(1-y_batch/1-output)))/batch_size
			for k in range(last-1, -1, -1):
				# dp1 = np.sum(np.multiply(weights[k+1].T, deltas[k+1]).T, axis=1).T
				dp1 = np.sum(np.multiply(weights[k+1].T, deltas[k+1]), axis=1).T
				# print(dp1)
				dp1.reshape(dp1.size)
				# print(np.multiply(weights[k+1].T, deltas[k+1]))
				# dp2 = np.sum(np.multiply(layers[k], 1-layers[k]), axis=1) #Uses Sigmoid Units (CHANGE FOR RELU)
				dp2 = np.multiply(layers[k], 1-layers[k]) #Uses Sigmoid Units (CHANGE FOR RELU)
				deltas[k] = np.multiply(dp1, dp2)

			#Calculate weight updates now
			for k in range(last, -1, -1):
				if (k > 0):
					# x_d = np.sum(layers[k-1], axis=1)
					x_d = layers[k-1]
				else:
					x_d = x
				# print(k)
				# print(x)
				# print((deltas[k]))
				# print((deltas[k].T))
				# if k != last:
					# addendum = np.matmul(x.reshape((len(x), 1)), deltas[k].reshape((1, len(deltas[k]))))
				addendum = np.matmul(deltas[k].reshape((deltas[k].size, 1)), x_d.reshape((1, x_d.size)))
				# else:
				# 	addendum = x*deltas[k].T
				weights[k] = weights[k] + eta*addendum
		n_iter+=1
		
		# if n_iter >=2:
		# 	cf1 = cost_function_arr[len(cost_function_arr)-1]
		# 	cf2 = cost_function_arr[len(cost_function_arr)-2]
		# 	if((math.fabs(cf1 - cf2) < err) & (cf1 < cf2)):
		# 		break
		print(('Cost Function - '+str(forward_prop(X, y, weights))))
		print('Train Accuracy - '+str(accuracy(X, y, weights)))
		print('Test Accuracy - '+str(accuracy(X_test, y_test, weights)))
		cost_function_arr.append(forward_prop(X, y, weights))
		train_accuracy_arr.append(accuracy(X, y, weights))
		test_accuracy_arr.append(accuracy(X_test, y_test, weights))
	return cost_function_arr, train_accuracy_arr, test_accuracy_arr, weights


X = np.genfromtxt('toy_data/toy_trainX.csv', delimiter=',')
y = np.genfromtxt('toy_data/toy_trainY.csv', delimiter=',')
print(y)

X_test = np.genfromtxt('toy_data/toy_testX.csv', delimiter=',')
y_test = np.genfromtxt('toy_data/toy_testY.csv', delimiter=',')

clf = LogisticRegression()
clf = clf.fit(X, y)
acc = clf.score(X, y)
print('Accuracy - '+str(acc))
acc = clf.score(X_test, y_test)
print('Accuracy - '+str(acc))

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

layers_dim = [2, 5, 1]
cf, tr, te, weights = train_network_single_eg(X, y, X_test, y_test, layers_dim, len(X))
plot_decision_boundary(return_output, X, y, 'contour_train_5.png')
plot_decision_boundary(return_output, X_test, y_test, 'contour_test_')
		
# for hidden_layer in [10, 20, 40, 5, 1, 2, 3]:
# 	layers_dim = [2]+[hidden_layer]+[1]
# 	cf, tr, te, weights = train_network_single_eg(X, y, X_test, y_test, layers_dim, len(X))
# 	plot_decision_boundary(return_output, X, y, 'contour_train_'+str(hidden_layer)+'.png')
# 	plot_decision_boundary(return_output, X_test, y_test, 'contour_test_'+str(hidden_layer)+'.png')
	
# 	plt.figure()
# 	x = np.arange(1, len(tr)+1)
# 	plt.plot(x, tr, 'b-', label='Training Acc')
# 	plt.plot(x, te, 'k-', label='Testing Acc')
# 	plt.legend()
# 	plt.savefig('accuracy_'+str(hidden_layer)+'.png')

# 	