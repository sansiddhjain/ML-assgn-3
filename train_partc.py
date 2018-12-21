from __future__ import division
import numpy as np
import tree_partc
import math
import statistics
import copy
import matplotlib.pyplot as plt

from read_data import preprocess

filename = "dtree_data/train.csv"
data = preprocess(filename)
train_data = copy.deepcopy(data)

filename = "dtree_data/test.csv"
test_data = preprocess(filename)

filename = "dtree_data/valid.csv"
valid_data = preprocess(filename)

print "Read and pre-processed data."

attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]
is_it_numeric = [False]*14
for i in range(14):
	if i+1 in [1, 3, 5, 11, 12, 13]:
		is_it_numeric[i] = True
		
attributes_idx = np.arange(1, 15)
unq_cls = {}

for i in attributes_idx:
	arr = np.unique(data[:, i:i+1])
	unq_cls[i] = arr
	# print arr

def calc_inf_gain(data, attr, isNumeric):
	arr_vals = unq_cls[attr]
	# print arr_vals
	data_copy = copy.deepcopy(data)
	if isNumeric:
		med = statistics.median(data_copy[:, attr])
		data_copy[:, attr] = list(map(float, data_copy[:, attr] > med))
	denom = data_copy.shape[0]
	total_no_of_vals = len(arr_vals)
	prob_vec = np.zeros(total_no_of_vals)
	Hy_vec = np.zeros(total_no_of_vals)

	# print sum(data_copy[:, 0] == 0)
	p0 = sum(data_copy[:, 0] == 0)/len(data_copy)
	p1 = sum(data_copy[:, 0] == 1)/len(data_copy)
	# print p0
	# print p1
	if ((p0 == 0) or (p1 == 0)):
		return 0
	Hy = -p0*np.log(p0)-p1*np.log(p1)

	for j in arr_vals:
		idx = (data_copy[:, attr] == j)
		prob_vec[j] = sum(idx)/denom
		
		subset = data_copy[idx, :]
		if (len(subset) == 0):
			Hy_vec[j] = 0
			continue
		p0 = sum(subset[:, 0] == 0)/len(subset)
		p1 = sum(subset[:, 0] == 1)/len(subset)
		# print '('+str(j)+', '+str(p0)+', '+str(p1)+')'
		if p0 == 0:
			surp0 = 0
		else:
			surp0 = -p0*np.log(p0)

		if p1 == 0:
			surp1 = 0
		else:
			surp1 = -p1*np.log(p1)

		Hy_vec[j] = surp0+surp1

	# print prob_vec
	# print Hy_vec
	return Hy - (np.dot(prob_vec, Hy_vec))

def choose_node_to_split(data):
	p0 = sum(data[:, 0] == 0)/len(data)
	p1 = sum(data[:, 0] == 1)/len(data)
	if ((p0 == 0) or (p1 == 0)):
		return 0
	information_gain_list = [calc_inf_gain(data, j, is_it_numeric[j-1]) for j in attributes_idx]
	# print information_gain_list
	node = np.argmax(np.asarray(information_gain_list)) + 1 # +1 Because all indices start with 1, not 0
	if (sum(np.asarray(information_gain_list) == 0) == len(information_gain_list)):
		# print data
		# print information_gain_list
		return 0
	else:
		return node

def count_nodes(node):
	if node.isChild:
		return 1
	else:
		res = 0
		for i in range(len(node.children)):
			res += count_nodes(node.children[i])
		return res + 1

#Traverses an already built tree
def traverse_tree(entry, node):
	if (node.isChild == False):
		attrib = node.attr
		# print attrib
		val = entry[attrib]
		if is_it_numeric[attrib-1]:
			splt = node.spltPt
			if val <= splt:
				return traverse_tree(entry, node.children[0])
			else:
				return traverse_tree(entry, node.children[1])
		else:
			return traverse_tree(entry, node.children[val])
	else:
		# print 'value ' + str(node.val)	
		return node.val

#Checks Accuracy
def check_acc(valid_data):
	pred_vals = np.zeros(valid_data.shape[0])
	for i in range(valid_data.shape[0]):
		example = valid_data[i, :]
		res = int(traverse_tree(example, root))
		pred_vals[i] = res
	
	acc = sum(pred_vals == valid_data[:, 0])/len(pred_vals)
	return acc

# print calc_inf_gain(data, 2)
# choose_node_to_split(data)

n_nodes = []
train_acc_arr = []
valid_acc_arr = []
test_acc_arr = []
#Build Entire Tree
def build_tree(node, data, level):
	attr = choose_node_to_split(data)
	if (attr == 0):
		#Change node to Leaf Node
		if (sum(data[:, 0] == 0) > sum(data[:, 0] == 1)):
			node.change_node_to_leaf(0)
			print 'Created Leaf Node with value 0. Currently at level '+str(level)+'. Data size - '+str(len(data))
		else:
			node.change_node_to_leaf(1)
			print 'Created Leaf Node with value 1. Currently at level '+str(level)+'. Data size - '+str(len(data))
		# if ((math.fabs(sum(data[:, 0] == 0) - sum(data[:, 0] == 1)) < len(data)) & (len(data > 0))):
		# 	print "DAMN Son."
			# print data
	else:
		node.assign_attribute(attr)
		if (sum(data[:, 0] == 0) > sum(data[:, 0] == 1)):
			node.assign_value(0)
			parent_val = 0
		else:
			node.assign_value(1)
			parent_val = 1
		print 'Created Node with attribute '+str(attr)+'. Currently at level '+str(level)+'. Data size - '+str(len(data))

		if not is_it_numeric[attr-1]:
			arr = unq_cls[attr]
			node.create_children(len(arr))
			n_nodes.append(count_nodes(root))
			train_acc_arr.append(check_acc(train_data))
			test_acc_arr.append(check_acc(test_data))
			valid_acc_arr.append(check_acc(valid_data))
			for i in range(len(arr)):
				if (sum(data[:, attr] == arr[i]) > 0):
					build_tree(node.children[i], data[(data[:, attr] == arr[i]), :], level+1)
				else:
					node.children[i].change_node_to_leaf(parent_val)
					print 'Created Leaf Node with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)
		else:
			med = statistics.median(data[:, attr])
			node.create_children(2)
			n_nodes.append(count_nodes(root))
			train_acc_arr.append(check_acc(train_data))
			test_acc_arr.append(check_acc(test_data))
			valid_acc_arr.append(check_acc(valid_data))
			if (sum(data[:, attr] <= med) > 0):
				build_tree(node.children[0], data[(data[:, attr] <= med), :], level+1)
			else:
				node.children[0].change_node_to_leaf(parent_val)
				print 'Created Leaf Node (Numeric) with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)

			if (sum(data[:, attr] > med) > 0):
				build_tree(node.children[1], data[(data[:, attr] > med), :], level+1)
			else:
				node.children[1].change_node_to_leaf(parent_val)
				print 'Created Leaf Node (Numeric) with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)

			node.is_numeric(True)
			node.assign_numeric_split(med)
	
print "Building Decision Tree (Part C)..."			
root = tree_partc.Node(0)
build_tree(root, data, 0)
print "Trained Decision Tree (Part C)!"

plt.figure()
plt.plot(n_nodes, train_acc_arr, 'b-', label='Training Acc')
plt.plot(n_nodes, valid_acc_arr, 'r-', label='Valid Acc')
plt.plot(n_nodes, test_acc_arr, 'k-', label='Testing Acc')
plt.xlabel('Number of Nodes')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('partc_accuracy.png')
print "Plotted Curves!"

train_acc = check_acc(train_data)
valid_acc = check_acc(valid_data)
test_acc = check_acc(test_data)
print 'Training Accuracy for Part (C) - '+str(train_acc)
print 'Validation Accuracy for Part (C) - '+str(valid_acc)
print 'Test Accuracy for Part (C) - '+str(test_acc)

# filename = "dtree_data/test.csv"
# test_data = preprocess(filename)

# pred_vals = np.zeros(test_data.shape[0])
# for i in range(test_data.shape[0]):
# 	example = test_data[i, :]
# 	res = int(traverse_tree(example, root))
# 	pred_vals[i] = res
# 	# print ''

# print pred_vals
# print test_data[:, 0]
# print sum(pred_vals == test_data[:, 0])
# acc = sum(pred_vals == test_data[:, 0])/len(pred_vals)
# print 'Accuracy - '+str(acc)

# filename = "dtree_data/valid.csv"
# test_data = preprocess(filename)

# pred_vals = np.zeros(test_data.shape[0])
# for i in range(test_data.shape[0]):
# 	example = test_data[i, :]
# 	res = int(traverse_tree(example, root))
# 	pred_vals[i] = res
# 	# print ''

# print pred_vals
# print test_data[:, 0]
# print sum(pred_vals == test_data[:, 0])
# acc = sum(pred_vals == test_data[:, 0])/len(pred_vals)
# print 'Accuracy - '+str(acc)
