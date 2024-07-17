import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

import adjustment
import mSBD
import graph

# Reference: Amin (https://causalai.net/r35.pdf) (Lemma 1)
def check_Tian_criterion(G,X):
	"""
	Check if X = De_{G(S_X)}(X) where S_X is an union of the c-components of X 

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.

	Returns:
	bool: True if Tian's criterion is satisfied 
	"""

	# 1. Union of S_X 
	S_X = graph.find_c_components(G, X)
	De_S_X = set( graph.find_descendant( graph.subgraphs(G,S_X), X) ) 
	if set(De_S_X) == set(X):
		return True 
	else:
		return False

def check_Generalized_Tian_criterion(G,X):
	"""
	Check if X = De_{G(S_X)}(X) where S_X is an union of the c-components of X 

	Parameters:
	G (nx.DiGraph): The directed graph.
	X (list): List of treatment variables.

	Returns:
	bool: True if Tian's criterion is satisfied 
	"""

	# 1. Union of S_X 
	for Xi in X: 
		Ch_Xi_except_X = list( set( graph.find_children(G, [Xi]) ) - set(X) )
		S_Xi = graph.find_c_components(G, [Xi])
		S_Ch_Xi = graph.find_c_components(G, Ch_Xi_except_X)
		if len(set(S_Xi).intersection(set(S_Ch_Xi))) > 0:
			return False
	return True 


def generalized_Tian_estimand(G, X, Y, latex, topo_V = None):
	if topo_V == None:
		topo_V = graph.find_topological_order(G)

	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	

	Q_V_SX = ""
	for i in range(len(V_SX)):
		Vi = V_SX[i]
		V_prev_i = ','.join(topo_V[:topo_V.index(Vi)]) 
		if len(V_prev_i) > 0:
			Q_V_SX += f"P({Vi} | {V_prev_i})"
		else:
			Q_V_SX += f"P({Vi})"

	idx = 0 
	X_copy = X[:]

	Q_SX = ""
	while len(X_copy) > 0: 
		Xi = X[idx]
		S_Xi = sorted( graph.find_c_components(G, [Xi]), key=lambda x: topo_V.index(x) )
		X_Ci = sorted( list(set(S_Xi).intersection(set(X))), key=lambda x: topo_V.index(x) )
		X_Ci_val = ','.join(char.lower() for char in X_Ci)
		X_copy = list(set(X_copy) - set(X_Ci))

		idx += 1

		if len(S_Xi) > 1: 
			Q_SXi_component = ""

			range_limit = len(S_Xi) if S_Xi[-1] not in X_Ci else len(S_Xi) - next((i for i, x in enumerate(reversed(S_Xi), 1) if not x.startswith('X')), len(S_Xi)) + 1
			if range_limit != len(S_Xi):
				last_X_idx = next((i for i, x in enumerate(reversed(S_Xi), 1) if not x.startswith('X')), len(S_Xi)) - 1
				last_X = S_Xi[-last_X_idx:]
				X_Ci_remained = list(set(X_Ci) - set(last_X))
			else:
				X_Ci_remained = X_Ci[:]
			

			for i in range(range_limit):
				Vi = S_Xi[i]
				V_prev_i = ','.join(topo_V[:topo_V.index(Vi)]) 
				if len(V_prev_i) > 0:
					Q_SXi_component += f"P({Vi} | {V_prev_i})"
				else:
					Q_SXi_component += f"P({Vi})"

			if len(X_Ci_remained) > 0:
				X_Ci_remained_val = ','.join(char.lower() for char in X_Ci_remained)
				if not latex:
					Q_SX += f"(\u03A3_{{{X_Ci_remained_val}}}{Q_SXi_component})"
				else:
					Q_SX += f"\\left(\\sum_{{{X_Ci_remained_val}}}{Q_SXi_component}\\right)"

			else:
				if not latex:
					Q_SX += f"({Q_SXi_component})"
				else:
					Q_SX += f"\\left({Q_SXi_component}\\right)"
				

	V_XY_lower = ', '.join(char.lower() for char in V_XY)
	
	if len(V_XY) > 0:
		if not latex:
			estimand = f"\u03A3_{{{V_XY_lower}}}{Q_V_SX}{Q_SX}"
		else:
			estimand = f"\\sum_{{{V_XY_lower}}}{Q_V_SX}{Q_SX}"
	else:
		if not latex:
			estimand = f"{Q_V_SX}{Q_SX}"
		else:
			estimand = f"{Q_V_SX}{Q_SX}"
	return estimand 


def Tian_estimand(G, X, Y, latex, topo_V = None):
	if topo_V == None:
		topo_V = graph.find_topological_order(G)

	X = sorted(X, key = lambda x: topo_V.index(x))
	SX = sorted( graph.find_c_components(G, X), key=lambda x: topo_V.index(x) )	
	V_SX = sorted( list( set(topo_V) - set(SX) ) , key=lambda x: topo_V.index(x) )	
	V_XY = sorted( list( set(topo_V) - set(X + Y)), key=lambda x: topo_V.index(x) )	
	V_XY_lower = ', '.join(char.lower() for char in V_XY)
	X_lower_val = ', '.join(char.lower() for char in X)

	Q_V_SX = ""
	for i in range(len(V_SX)):
		Vi = V_SX[i]
		V_prev_i = ','.join(topo_V[:topo_V.index(Vi)]) 
		if len(V_prev_i) > 0:
			Q_V_SX += f"P({Vi} | {V_prev_i})"
		else:
			Q_V_SX += f"P({Vi})"

	Q_SX = ""
	range_limit = len(SX) if SX[-1] not in X else len(SX) - next((i for i, x in enumerate(reversed(SX), 1) if not x.startswith('X')), len(SX)) + 1
	if range_limit != len(SX):
		last_X_idx = next((i for i, x in enumerate(reversed(SX), 1) if not x.startswith('X')), len(SX)) - 1
		last_X = SX[-last_X_idx:]
		X_remained = list(set(X) - set(last_X))
	else:
		X_remained = X[:]

	for i in range(range_limit):
		Vi = SX[i]
		V_prev_i = ','.join(topo_V[:topo_V.index(Vi)]) 
		if len(V_prev_i) > 0:
			Q_SX += f"P({Vi} | {V_prev_i})"
		else:
			Q_SX += f"P({Vi})"

	if len(X_remained) > 0:
		X_remained = ','.join(char.lower() for char in X_remained)
		if not latex:
			Q_SX = f"(\u03A3_{{{X_remained}}}{Q_SX})"
		else:
			Q_SX = f"\\left(\\sum_{{{X_remained}}}{Q_SX}\\right)"

	else:
		if not latex:
			Q_SX = f"({Q_SX})"
		else:
			Q_SX = f"\\left({Q_SX}\\right)"



	if len(V_XY) > 0:
		if not latex:
			estimand = f"\u03A3_{{{V_XY_lower}}}{Q_V_SX}{Q_SX}"
		else:
			estimand = f"\\sum_{{{V_XY_lower}}}{Q_V_SX}{Q_SX}"
	else:
		if not latex:
			estimand = f"{Q_V_SX}{Q_SX}"
		else:
			estimand = f"{Q_V_SX}{Q_SX}"
	return estimand 