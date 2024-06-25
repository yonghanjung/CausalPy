import pyperclip
import graph
import identify
import examples

if __name__ == "__main__":
	''' Predefined examples '''
	# [graph_dict, node_positions, X, Y] = examples.Tikka1() 
	# [graph_dict, node_positions, X, Y] = examples.Tikka2() 
	# [graph_dict, node_positions, X, Y] = examples.FD0() 
	# [graph_dict, node_positions, X, Y] = examples.FD1() 
	# [graph_dict, node_positions, X, Y] = examples.FD2() # answer: {A,B,C,D}, minimal = {D}
	# [graph_dict, node_positions, X, Y] = examples.FD3() # answer: {A,B,C,D}, minimal = {A}
	# [graph_dict, node_positions, X, Y] = examples.FD4() # answer: {A,B,C,D,E}, minimal = {BDE} or Z={B}, C={DE}
	# [graph_dict, node_positions, X, Y] = examples.FD5() # answer: {A,B,D}, minimal = {A,D} or Z={A}, C={D}
	# [graph_dict, node_positions, X, Y] = examples.BD_vdZ() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum2() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum3() 
	# [graph_dict, node_positions, X, Y] = examples.Tian() 
	# [graph_dict, node_positions, X, Y] = examples.Tian2() 
	# [graph_dict, node_positions, X, Y] = examples.Tian3() 
	# [graph_dict, node_positions, X, Y] = examples.Tian_mSBD() 
	# [graph_dict, node_positions, X, Y] = examples.mSBD1() 
	# [graph_dict, node_positions, X, Y] = examples.mSBD2()
	# [graph_dict, node_positions, X, Y] = examples.mSBD3()
	# [graph_dict, node_positions, X, Y] = examples.mSBD_minimum() 
	# [graph_dict, node_positions, X, Y] = examples.Napkin_FD() 
	# [graph_dict, node_positions, X, Y] = examples.Napkin() 
	# [graph_dict, node_positions, X, Y] = examples.Double_Napkin() 
	# [graph_dict, node_positions, X, Y] = examples.Chris_FD1() 
	# [graph_dict, node_positions, X, Y] = examples.unID1() 
	# [graph_dict, node_positions, X, Y] = examples.unID2() 

	# Generate the random graph 
	[graph_dict, node_positions, X, Y] = examples.Random_Example_Generator(num_observables = 10, num_unobservables = 5, num_treatments = 1, num_outcomes = 1, 
																			condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = True)
	
	# Copy the graph for comparing with Fusion
	pyperclip.copy(graph.graph_dict_to_fusion_graph(graph_dict))
	
	# Get the graph from the graph_dict
	G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)
	
	# Visualize the graph 
	graph.visualize(G)

	# Identify the causal effect P(Y | do(X)) from G 
	identify.causal_identification(G,X,Y)

	# Draw the C-tree and AC-tree 
	# identify.draw_C_tree(G,X,Y)
	# identify.draw_AC_tree(G,X,Y)
	




	

