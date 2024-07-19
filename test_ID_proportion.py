import pyperclip
import graph
import identify
import examples
import random_generator
import SCM 
import adjustment
import frontdoor
import mSBD
import tian 


if __name__ == "__main__":
	# Generate the random graph 
	[graph_dict, node_positions, X, Y] = random_generator.Random_Graph_Generator(num_observables = 20, num_unobservables = 3, num_treatments = 2, num_outcomes = 1, 
																			condition_ID = True, condition_BD = False, condition_mSBD = True, condition_FD = False, condition_Tian = False, condition_gTian = False)
	# graph_dict = {'U_V1_X3': ['V1', 'X3'], 'V1': ['X1'], 'U_V1_X2': ['X2', 'V1'], 'X2': ['X1', 'V1', 'V2', 'Y1'], 'U_V1_Y1': ['Y1', 'V1'], 'Y1': [], 'X3': ['X2', 'X1', 'V1'], 'U_X1_X3': ['X3', 'X1'], 'X1': ['V2'], 'V2': ['Y1']}
	# X = ['X1','X2']; Y = ['Y1']; node_positions = None
	G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)

	# Generate the random SCM 
	# [scm, X, Y] = random_generator.Random_SCM_Generator(num_observables = 5, num_unobservables = 3, num_treatments = 2, num_outcomes = 1, 
	# 																		condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False, condition_Tian = False, condition_gTian = True)
	# sample_data = scm.generate_samples(10000)[topo_V]
	# print(sample_data)
	# G = scm.graph
	

	# Visualize the graph 
	# graph.visualize(G)
	

	# Identify the causal effect P(Y | do(X)) from G 
	# G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)
	print( identify.causal_identification(G,X,Y, False) )


	# Draw the C-tree and AC-tree 
	# identify.draw_C_tree(G,X,Y)
	# identify.draw_AC_tree(G,X,Y)

	# Copy the graph for comparing with Fusion
	# pyperclip.copy(graph.graph_dict_to_fusion_graph(graph_dict))
	




	

