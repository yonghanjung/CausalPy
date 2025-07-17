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
import random
import numpy as np 


if __name__ == "__main__":
	# Generate the random graph 
	base_seednum = 190602 
	np.random.seed(base_seednum)
	random.seed(base_seednum)

	num_sim = 10000

	num_observables = random.randint(5, 15)  # A random integer between 1 and 10
	num_unobservables = random.randint(0, num_observables)
	num_treatments = random.randint(1, int(round(num_observables/2)))
	# num_treatments = 2
	num_outcomes = 1

	graph_counter = 0 
	graph_type = {"BD": 0, "mSBD": 0, "FD": 0, "Tian": 0, "gTian": 0, "Product": 0, "Ratio": 0, "MECE_FD": 0, "MECE_Tian": 0, "MECE_gTian": 0, "MECE_product": 0}

	while graph_counter < num_sim: 
		seednum = random.randint(1, 1000000)
		[graph_dict, node_positions, X, Y] = random_generator.Random_Graph_Generator2(num_observables = num_observables, num_unobservables = num_unobservables, num_treatments = num_treatments, num_outcomes = num_outcomes, condition_ID = True, seednum = seednum)
		G = graph.create_acyclic_graph(graph_dict=graph_dict, an_Y_graph_TF = False, Y = None, node_positions = node_positions)

		G, X, Y = identify.preprocess_GXY_for_ID(G, X, Y)

		satisfied_BD = adjustment.check_admissibility(G, X, Y)
		satisfied_mSBD = mSBD.constructive_SAC_criterion(G, X, Y)
		satisfied_FD = frontdoor.constructive_FD(G, X, Y)
		satisfied_Tian = tian.check_Tian_criterion(G, X)
		satisfied_gTian = tian.check_Generalized_Tian_criterion(G, X)
		satisfied_product = tian.check_product_criterion(G, X, Y)

		if satisfied_BD: 
			graph_type['BD'] += 1

		if satisfied_mSBD:
			graph_type['mSBD'] += 1

		if satisfied_FD: 
			graph_type['FD'] += 1

		if satisfied_Tian: 
			graph_type['Tian'] += 1

		if satisfied_gTian:
			graph_type['gTian'] += 1

		if satisfied_product:
			graph_type['Product'] += 1

		if satisfied_BD == False and satisfied_mSBD == False and satisfied_FD:
			graph_type['MECE_FD'] += 1

		if satisfied_BD == False and satisfied_mSBD == False and satisfied_FD == False and satisfied_Tian:
			graph_type['MECE_Tian'] += 1

		if satisfied_BD == False and satisfied_mSBD == False and satisfied_FD == False and satisfied_Tian == False and satisfied_gTian:
			graph_type['MECE_gTian'] += 1

		if satisfied_BD == False and satisfied_mSBD == False and satisfied_FD == False and satisfied_Tian == False and satisfied_gTian == False and satisfied_product:
			graph_type['MECE_product'] += 1

		if satisfied_BD == False and satisfied_mSBD == False and satisfied_Tian == False and satisfied_gTian == False and satisfied_product == False:
			graph_type['Ratio'] += 1

		graph_counter += 1 
		print(graph_counter)

	print(graph_type)



	

