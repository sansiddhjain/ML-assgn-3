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