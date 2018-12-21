class Node(object):
	"""docstring for Node"""
	def __init__(self, val):
		self.isChild = True
		self.val = val
	
	def assign_attribute(self, attr):
		self.attr = attr
		self.isChild = False

	def assign_value(self, val):
		self.val = val
		
	def create_children(self, n_children):
		children = [Node(self.val) for i in range(n_children)]
		self.children = children

	def change_node_to_leaf(self, val):
		self.isChild = True
		self.val = val

	def change_leaf_to_node(self, val):
		self.isChild = False
		self.val = val