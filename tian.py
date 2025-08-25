import scipy.stats as stats
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations

import identify
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

def check_product_criterion(G,X,Y):
	adj_dict_components, adj_dict_operations = identify.return_AC_tree(G, X, Y)
	for adj_dict_component in adj_dict_components.values():
		if len(adj_dict_component) > 1:
			return False 
	return True 

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
		S_Xi = graph.find_c_components(G, [Xi]) # c-partition
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

'''
Tian's adjustment
1. topo_V
2. 
'''

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

# --- dc-partition (Def. 9) and generalized Tian check (Def. 10) ---

def _as_list(obj):
    """Utility: normalize scalars/iterables to a list of strings."""
    if obj is None:
        return []
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return [obj]

def _c_partition_of_X(G, X):
    """
    Partition X by c-components in G.
    Returns a list of sets [Xc1, Xc2, ...], each wholly contained in a single c-component.
    """
    X = _as_list(X)
    buckets = {}  # key: frozenset(c-comp nodes), value: set of X in that c-comp
    for xi in X:
        cc_nodes = set(graph.find_c_components(G, [xi]))  # union of the c-component containing xi
        key = frozenset(cc_nodes)
        if key not in buckets:
            buckets[key] = set()
        buckets[key].add(xi)
    # Return just the X-contents for each c-component; ignore empty
    return [sorted(list(s)) for s in buckets.values() if len(s) > 0], [set(k) for k in buckets.keys()]

def dc_partition(G, X, Y):
    r"""
    Definition 9 (Descendant-closed c-partition). For each c-part X_c ⊆ X,
    let
        X'_c = De_{G(c(X_c))}(X_c) \ PCP(X, Y) \ Y,
    where c(X_c) is the c-component containing X_c, G(c(X_c)) is the induced subgraph on that component,
    De_{G(c(X_c))}(X_c) are directed descendants *within that subgraph*,
    and PCP(X,Y) is the proper causal path set. (Defs. and notation per the draft.)  # :contentReference[oaicite:1]{index=1}

    Returns:
        dc_blocks: list of lists, each is X'_c (sorted)
        meta: list of dicts with extra info per block (c-comp nodes, etc.)
    """
    X = _as_list(X)
    Y = set(_as_list(Y))

    # c-partition (groups of X by c-component)
    Xc_list, ccomp_nodes_list = _c_partition_of_X(G, X)

    pcp = set(adjustment.proper_causal_path(G, X, list(Y)))  # PCP(X,Y)

    dc_blocks = []
    meta = []

    for Xc, ccomp_nodes in zip(Xc_list, ccomp_nodes_list):
        # G(c(Xc)): subgraph induced by c-component nodes (graph.subgraphs keeps relevant U's too)
        Gc = graph.subgraphs(G, list(ccomp_nodes))

        # De_{G(c(Xc))}(Xc) (observable descendants within the subgraph; the helper includes the nodes themselves)
        De_in_c = set(graph.find_descendant(Gc, Xc))

        # X'_c = De_{G(c(Xc))}(Xc) \ PCP(X,Y) \ Y
        Xc_prime = list(sorted(list(De_in_c - pcp - Y)))

        dc_blocks.append(Xc_prime)
        meta.append({
            "Xc": list(sorted(Xc)),
            "c_component": list(sorted(ccomp_nodes)),
            "De_in_c": list(sorted(De_in_c)),
            "PCP": list(sorted(pcp)),
            "Y": list(sorted(Y)),
            "Xc_prime": list(sorted(Xc_prime)),
        })

    return dc_blocks, meta

def check_dcGenTian(G, X, Y, return_witness=False):
    r"""
    Definition 10 (Generalized Tian’s Criterion via dc-partition).
    Let {X'_1, ..., X'_c} = dc_partition(G, X, Y).
    X satisfies the generalized Tian criterion iff, for every nonempty X'_i,
        X'_i = De_{G(c(X'_i))}(X'_i).
    (All notation as in the draft.)  # :contentReference[oaicite:2]{index=2}

    Args:
        G: nx.DiGraph
        X: list[str] | str
        Y: list[str] | str
        return_witness: if True, return (bool, details) with a failing witness

    Returns:
        bool  |  (bool, dict) if return_witness
    """
    dc_blocks, meta = dc_partition(G, X, Y)

    for block, info in zip(dc_blocks, meta):
        Xp = set(block)
        if len(Xp) == 0:
            continue  # vacuously satisfied for this block
        # c(X'_i): c-component containing X'_i (they should all be within a single c-comp)
        c_nodes = set(graph.find_c_components(G, list(Xp)))
        Gc = graph.subgraphs(G, list(c_nodes))
        De_in_c = set(graph.find_descendant(Gc, list(Xp)))  # includes Xp itself

        if De_in_c != Xp:
            if return_witness:
                witness = {
                    "X_prime": sorted(list(Xp)),
                    "c_component_of_X_prime": sorted(list(c_nodes)),
                    "De_in_c_of_X_prime": sorted(list(De_in_c)),
                    "meta": info,
                }
                return False, witness
            return False

    return (True, {"dc_blocks": dc_blocks, "meta": meta}) if return_witness else True







