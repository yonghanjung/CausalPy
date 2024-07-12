import pyperclip
import graph
import identify
import examples
import SCM 
import adjustment
import frontdoor
import mSBD
import tian 

if __name__ == "__main__":
	''' Predefined examples '''
	# [graph_dict, node_positions, X, Y] = examples.Tikka1() 
	# [graph_dict, node_positions, X, Y] = examples.Tikka2() 
	# [graph_dict, node_positions, X, Y] = examples.FD0() 
	# [graph_dict, node_positions, X, Y] = examples.FD1() 
	# [graph_dict, node_positions, X, Y] = examples.FD2() # answer: {A,B,C,D}, minimal = {D} or {A}
	# [graph_dict, node_positions, X, Y] = examples.FD3() # answer: {A,B,C,D}, minimal = {A}
	# [graph_dict, node_positions, X, Y] = examples.FD4() # answer: {A,B,C,D,E}, minimal = {BDE} or {ADE} | Z={B} or {A}, C={DE}
	# [graph_dict, node_positions, X, Y] = examples.FD5() # answer: {A,B,D}, minimal = {A,D} or Z={A}, C={D}
	# [graph_dict, node_positions, X, Y] = examples.UCA1() 
	# [graph_dict, node_positions, X, Y] = examples.UCA2() 
	# [graph_dict, node_positions, X, Y] = examples.BD_vdZ() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum2() 
	# [graph_dict, node_positions, X, Y] = examples.BD_minimum3() 
	# [graph_dict, node_positions, X, Y] = examples.Tian1() 
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
	# [graph_dict, node_positions, X, Y] = examples.Rina1() 
	
	# Generate the random graph 
	[graph_dict, node_positions, X, Y] = examples.Random_Example_Generator(num_observables = 6, num_unobservables = 3, num_treatments = 2, num_outcomes = 2, 
																			condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False, condition_Tian = False, condition_gTian = True)
	# graph_dict = {'U_V1_Y2': ['V1', 'Y2'], 'V1': ['V2', 'Y1', 'X2', 'X1'], 'U_Y1_X2': ['Y1', 'X2'], 'Y1': ['Y2'], 'U_X2_Y1': ['X2', 'Y1'], 'X2': ['X1', 'Y2', 'V2'], 'Y2': [], 'X1': ['Y1', 'Y2'], 'V2': ['Y1', 'X1', 'Y2']}
	# X = ['X1','X2','X3']; Y = ['Y1','Y2']; node_positions = None

	G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)

	# Generate the random SCM 
	# [scm, X, Y] = examples.Random_SCM_Generator(num_observables = 6, num_unobservables = 5, num_treatments = 2, num_outcomes = 1, 
	# 										condition_ID = True, condition_BD = False, condition_mSBD = False, condition_FD = False)
	# sample_data = scm.generate_samples(10000)
	# print(sample_data)
	# G = scm.graph
	

	# Visualize the graph 
	# graph.visualize(G)
	

	# Identify the causal effect P(Y | do(X)) from G 
	print( identify.causal_identification(G,X,Y, False) )

	# Draw the C-tree and AC-tree 
	# identify.draw_C_tree(G,X,Y)
	# identify.draw_AC_tree(G,X,Y)

	# Copy the graph for comparing with Fusion
	pyperclip.copy(graph.graph_dict_to_fusion_graph(graph_dict))
	




	

