from __future__ import division
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import statistics

from read_data import preprocess
import tree
import tree_partc

filename = "dtree_data/train.csv"
data = preprocess(filename)
train_data = copy.deepcopy(data)

filename = "dtree_data/test.csv"
test_data = preprocess(filename)

filename = "dtree_data/valid.csv"
valid_data = preprocess(filename)

print "Read and pre-processed data."

attributes = ["rich","age","wc","fnlwgt","edu","edun","mar","occ","rel","race","sex","capg","canpl","hpw","nc"]
attributes_idx = np.arange(1, 15)
unq_cls = {}
is_it_numeric = [False]*14
for i in range(14):
	if i+1 in [1, 3, 5, 11, 12, 13]:
		is_it_numeric[i] = True


#---------------PART(A)--------------------

for i in attributes_idx:
	arr = np.unique(data[:, i])
	unq_cls[i] = arr
	
#Calculates Information Gain at a particular node with respect to a particular attribute
def calc_inf_gain(data, attr):
	arr_vals = unq_cls[attr]
	denom = data.shape[0]
	total_no_of_vals = len(arr_vals)
	prob_vec = np.zeros(total_no_of_vals)
	Hy_vec = np.zeros(total_no_of_vals)

	p0 = sum(data[:, 0] == 0)/len(data)
	p1 = sum(data[:, 0] == 1)/len(data)
	if ((p0 == 0) or (p1 == 0)):
		return 0
	Hy = -p0*np.log(p0)-p1*np.log(p1)

	for j in arr_vals:
		idx = (data[:, attr] == j)
		prob_vec[j] = sum(idx)/denom
		subset = data[idx, :]
		
		if (len(subset) == 0):
			Hy_vec[j] = 0
			continue
		p0 = sum(subset[:, 0] == 0)/len(subset)
		p1 = sum(subset[:, 0] == 1)/len(subset)
	
		if p0 == 0:
			surp0 = 0
		else:
			surp0 = -p0*np.log(p0)

		if p1 == 0:
			surp1 = 0
		else:
			surp1 = -p1*np.log(p1)

		Hy_vec[j] = surp0+surp1

	return Hy - (np.dot(prob_vec, Hy_vec))

#Calculates the node to split on, on the basis of maximum information gain
def choose_node_to_split(data):
	p0 = sum(data[:, 0] == 0)/len(data)
	p1 = sum(data[:, 0] == 1)/len(data)
	if ((p0 == 0) or (p1 == 0)):
		return 0
	information_gain_list = [calc_inf_gain(data, j) for j in attributes_idx]
	node = np.argmax(np.asarray(information_gain_list)) + 1 # +1 Because all indices start with 1, not 0
	if (sum(np.asarray(information_gain_list) == 0) == len(information_gain_list)):
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
		idx = entry[attrib]
		return traverse_tree(entry, node.children[idx])
	else:
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
			# print 'Created Leaf Node with value 0. Currently at level '+str(level)+'. Data size - '+str(len(data))
		else:
			node.change_node_to_leaf(1)
			# print 'Created Leaf Node with value 1. Currently at level '+str(level)+'. Data size - '+str(len(data))
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
		# print 'Created Node with attribute '+str(attr)+'. Currently at level '+str(level)+'. Data size - '+str(len(data))
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
				# print 'Created Leaf Node with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)

n_nodes = []
train_acc_arr = []
valid_acc_arr = []
test_acc_arr = []
#Build Entire Tree (For Part C)
def build_tree_partc(node, data, level):
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
					build_tree_partc(node.children[i], data[(data[:, attr] == arr[i]), :], level+1)
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
				build_tree_partc(node.children[0], data[(data[:, attr] <= med), :], level+1)
			else:
				node.children[0].change_node_to_leaf(parent_val)
				print 'Created Leaf Node (Numeric) with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)

			if (sum(data[:, attr] > med) > 0):
				build_tree_partc(node.children[1], data[(data[:, attr] > med), :], level+1)
			else:
				node.children[1].change_node_to_leaf(parent_val)
				print 'Created Leaf Node (Numeric) with value '+str(parent_val)+' because training set didn\'t have data for particular attribute. At level '+str(level+1)

			node.is_numeric(True)
			node.assign_numeric_split(med)
	

root = tree_partc.Node(0)
print "Building Decision Tree (Part C)..."
build_tree_partc(root, data, 0)
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

# n_nodes = []
# train_acc_arr = []
# valid_acc_arr = []
# test_acc_arr = []
# root = tree.Node(0)
# print "Building Decision Tree..."
# build_tree(root, data, 0)
# print "Trained Decision Tree!"

# plt.figure()
# plt.plot(n_nodes, train_acc_arr, 'b-', label='Training Acc')
# plt.plot(n_nodes, valid_acc_arr, 'r-', label='Valid Acc')
# plt.plot(n_nodes, test_acc_arr, 'k-', label='Testing Acc')
# plt.xlabel('Number of Nodes')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('parta_accuracy.png')
# print "Plotted Curves!"

# train_acc = check_acc(train_data)
# valid_acc = check_acc(valid_data)
# test_acc = check_acc(test_data)
# print 'Training Accuracy for Part (A) - '+str(train_acc)
# print 'Validation Accuracy for Part (A) - '+str(valid_acc)
# print 'Test Accuracy for Part (A) - '+str(test_acc)

# #---------------PART(B)--------------------

# #Greedily prunes tree
# n_nodes = []
# train_acc_arr = []
# valid_acc_arr = []
# test_acc_arr = []
# def prune_tree(node):
# 	if not node.isChild:
# 		val_arr = np.zeros(len(node.children))
# 		for i in range(len(node.children)):
# 			val_arr[i] = prune_tree(node.children[i])
# 		val = np.argmax(val_arr)
# 		node.change_node_to_leaf(node.val)
# 		acc = check_acc(valid_data)
# 		if (acc > val):
# 			n_nodes.append(count_nodes(root))
# 			train_acc_arr.append(check_acc(train_data))
# 			test_acc_arr.append(check_acc(test_data))
# 			valid_acc_arr.append(check_acc(valid_data))
# 			return acc
# 		else:
# 			node.change_leaf_to_node(node.val)
# 			return val
# 	else:
# 		return check_acc(valid_data)

# print "Beginning Pruning of Tree..."
# prune_tree(root)
# print "Finished Pruning!"

# x = np.arange(1, len(n_nodes)+1)
# plt.figure()
# plt.subplot(211)
# plt.plot(x, train_acc_arr, 'b-', label='Training Acc')
# plt.plot(x, valid_acc_arr, 'r-', label='Valid Acc')
# plt.plot(x, test_acc_arr, 'k-', label='Testing Acc')
# plt.xlabel('Pruning Step #')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.subplot(212)
# plt.plot(x, n_nodes, 'b-', label='# of Nodes')
# plt.xlabel('Pruning Step #')
# plt.ylabel('# of Nodes')
# plt.legend()
# plt.savefig('partb_accuracy.png')
# print "Plotted Curves!"

# train_acc = check_acc(train_data)
# valid_acc = check_acc(valid_data)
# test_acc = check_acc(test_data)
# print 'Training Accuracy for Part (B) - '+str(train_acc)
# print 'Validation Accuracy for Part (B) - '+str(valid_acc)
# print 'Test Accuracy for Part (B) - '+str(test_acc)