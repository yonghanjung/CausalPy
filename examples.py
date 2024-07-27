import scipy.stats as stats
import networkx as nx
import numpy as np 
import random 

import graph
import SCM
import adjustment
import identify
import mSBD
import frontdoor
import tian

def Verma2():
	graph_dict = {
		"U_XV2": ["V2","X"],
		"U_V2Y": ["V2","Y"],
		"U_V1V3": ["V1","V3"],
		"X": ["V1"],
		"V1": ["V2"],
		"V2": ["V3"],
		"V3": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_XV2": (100,10),
		"U_V2Y": (300,10),
		"U_V1V3": (200,-10),
		"X": (0,0),
		"V1": (100,0),
		"V2": (200,0),
		"V3": (300,0),
		"Y": (400,0)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]


def Tian1():
	graph_dict = {
		"U_V1X": ["V1","X"],
		"U_V1V3": ["V1","V3"],
		"U_V2V3": ["V2","V3"],
		"U_V3V5": ["V3","V5"],
		"U_V4V5": ["V4","V5"],
		"U_V1Y": ["V1","Y"],
		"V1": ["V2"],
		"V2": ["X"],
		"V3": ["V4"],
		"V4": ["X"],
		"V5": [],
		"X": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_V1X": (90,210),
		"U_V1V3": (291,277),
		"U_V2V3": (240,180),
		"U_V3V5": (330,180),
		"U_V4V5": (300,90),
		"U_V1Y": (420,270),
		"V1": (260,327),
		"V2": (180,177),
		"V3": (300,210),
		"V4": (240,120),
		"V5": (330,120),
		"X": (100,20),
		"Y": (460,20)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Tian2():
	graph_dict = {
		"U_V1X": ["V1","X"],
		"U_V1Y": ["V1","Y"],
		"U_V1V3": ["V1","V3"],
		# "U_V2V3": ["V2","V3"],
		"U_V2V6": ["V2","V6"],
		"U_V3V4": ["V3","V4"],
		"U_V3V7": ["V3","V7"],
		"U_V3V8": ["V3","V8"],
		"U_V4V7": ["V4","V7"],
		# "U_V6V7": ["V6","V7"],
		"U_V7V8": ["V7","V8"],
		"V1": ["V2"],
		"V2": ["V4"],
		"V4": ["V6"],
		"V6": ["X"],
		"X": ["Y"],
		"Y": [],
		"V3": ["V5"],
		"V5": ["V7"],
		"V7": ["X"],
		"V8": []
	}
	node_positions = {
		"U_V1X": (-50,-20),
		"U_V1Y": (90,-20),
		"U_V1V3": (17,-8),
		"U_V2V3": (10,-20),
		"U_V2V6": (-35,-35),
		"U_V3V4": (10,-30),
		"U_V3V7": (43,-42),
		"U_V3V8": (50,-30),
		"U_V4V7": (-4,-50),
		"U_V6V7": (-8,-60),
		"U_V7V8": (40,-50),
		"V1": (0,0),
		"V2": (-10,-20),
		"V4": (-20,-40),
		"V6": (-30,-60),
		"X": (-40,-80),
		"Y": (80,-80),
		"V3": (40,-20),
		"V5": (30,-40),
		"V7": (20,-60),
		"V8": (60,-40)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Tian3():
	graph_dict = {
		"U_V1X": ["V1","X"],
		"U_V1Y": ["V1","Y"],
		"U_V1V3": ["V1","V3"],
		"U_V2V6": ["V2","V6"],
		"U_V3V4": ["V3","V4"],
		"U_V3V7": ["V3","V7"],
		"U_V3V8": ["V3","V8"],
		"U_V4V7": ["V4","V7"],
		"U_V7V8": ["V7","V8"],
		"V1": ["V2"],
		"V2": ["V3","V4"],
		"V4": ["V6"],
		"V6": ["X"],
		"X": ["Y"],
		"Y": [],
		"V3": ["V7"],

		"V7": ["X", "V6"],
		"V8": []
	}
	node_positions = {
		"U_V1X": (-50,-20),
		"U_V1Y": (90,-20),
		"U_V1V3": (17,-8),
		"U_V2V6": (-35,-35),
		"U_V3V4": (10,-30),
		"U_V3V7": (43,-42),
		"U_V3V8": (50,-30),
		"U_V4V7": (-4,-50),
		"U_V7V8": (40,-50),
		"V1": (0,0),
		"V2": (-10,-20),
		"V4": (-20,-40),
		"V6": (-30,-60),
		"X": (-40,-80),
		"Y": (80,-80),
		"V3": (40,-20),
		"V7": (20,-60),
		"V8": (60,-40)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Tikka1():
	graph_dict = {
		"U_V2X": ["V2","X"],
		"U_V3X": ["V3","X"],
		"U_XY": ["X","Y"],
		"U_V2Y": ["V2","Y"],
		"V2": ["X", "V1", "V3"],
		"X": ["V1"],
		"V1": ["Y"],
		"V3": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_V2X": (100,50),
		"U_V3X": (57,57),
		"U_XY": (-30,-100),
		"U_V2Y": (150,-160),
		"V2": (200,0),
		"X": (0,0),
		"V1": (15,-100),
		"V3": (100,-150),
		"Y": (50,-200)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Tikka2():
	graph_dict = {
		"U_WX": ["W","X"],
		"U_WY": ["W","Y"],
		"W": ["X"],
		"X": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_WX": (50,75),
		"U_WY": (150,75),
		"W": (100,50),
		"X": (0,0),
		"Z": (100,0),
		"Y": (200,0)
	}
	X = ["X","Z","W"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Chris_FD1():
	graph_dict = {
		"U_XY": ["X","Y"],
		"U_XD": ["X","D"],
		"X": ["A"],
		"A": ["B", "C", "D"],
		"B": ["Y"],
		"C": ["Y"],
		"D": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_XY": (180,60),
		"U_XD": (90,-120),
		"X": (0,0),
		"A": (120,0),
		"B": (240,60),
		"C": (240,0),
		"D": (240,-60),
		"Y": (360,0)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def BD_minimum():
	graph_dict = {
		"Z1": ["X", "Z3"],
		"Z3": ["X","Y"],
		"Z2": ["Z3", "Y"],
		"X": ["Y"],
		"Y": []
	}
	node_positions = {
		"X": (0,0),
		"Y": (100,0),
		"Z3": (50,50), 
		"Z1": (0,100), 
		"Z2": (100,100)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def BD_minimum2():
	graph_dict = {
		"Z1": ["Z2","Z3"],
		"Z2": ["X"],
		"X": ["Y"],
		"Z3": ["X", "Y"],
		"Z4": ["Z3", "Z5"],
		"Z5": ["Y"],
		"Y": []
	}
	node_positions = {
		"X": (0,0),
		"Y": (100,0),
		"Z3": (50,50), 
		"Z2": (0,50), 
		"Z1": (0,100), 
		"Z5": (100,50),
		"Z4": (100,100)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def BD_minimum3():
	graph_dict = {
		"U": ["Z4", "Y"],
		"Z1": ["Z2","Z3"],
		"Z2": ["X"],
		"X": ["Y"],
		"Z3": ["X", "Y"],
		"Z4": ["Z3", "Z5"],
		"Z5": ["Y"],
		"Y": []
	}
	node_positions = {
		"X": (0,0),
		"Y": (100,0),
		"Z3": (50,50), 
		"Z2": (0,50), 
		"Z1": (0,100), 
		"Z5": (100,50),
		"Z4": (100,100),
		"U": (150,50)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def BD_vdZ():
	graph_dict = {
		"X1": ["Y1", "Z1"],
		"Z1": ["Z2"],
		"Z2": ["X2"],
		"Y2": ["Z2"],
		"Y1": [],
		"X2": []
	}
	node_positions = {
		"X1": (0,0),
		"Y1": (100,0),
		"Z1": (50,-50),
		"Z2": (30,-75),
		"X2": (0,-100),
		"Y2": (100,-100)
	}
	X = ["X1", "X2"]
	Y = ["Y1", "Y2"]

	return [graph_dict, node_positions, X, Y]

def Double_Napkin():
	graph_dict = {
		"U_V2V3": ["V2","V3"],
		"U_V1X": ["V1","X"],
		"U_V1Y": ["V1","Y"],
		"U_V3X": ["V3","X"],
		"V1": ["V2"],
		"V2": ["X"],
		"V3": ["V4"],
		"V4": ["V2"],
		"X": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_V2V3": (10,24),
		"U_V1X": (45,12),
		"U_V1Y": (100,20),
		"U_V3X": (-32,10),
		"V1": (50,30),
		"V2": (25,15),
		"V3": (-25,25),
		"V4": (0,20),
		"X": (0,0),
		"Y": (100,0)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Napkin():
	graph_dict = {
		"U_WX": ["W","X"],
		"U_WY": ["W","Y"],
		"W": ["R"],
		"R": ["X"],
		"X": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_WX": (10,150),
		"U_WY": (190,135),
		"W": (100,200),
		"R": (50,100),
		"X": (0,0),
		"Y": (200,0)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]


def mSBD1():
	graph_dict = {
		"U_X1Z": ["X1","Z"],
		"U_ZY": ["Z","Y"],
		"C": ["X1","Z","Y"],
		"Z": ["X2"],
		"X1": ["Z","X2","Y"],
		"X2": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_X1Z": (-35,-30),
		"U_ZY": (-24,-48),
		"C": (0,0),
		"Z": (-17,-20),
		"X1": (-45,-15),
		"X2": (-32,-50),
		"Y": (-32,-80)
	}
	X = ["X1", "X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def mSBD2():
	graph_dict = {
		"U_Z1Y2": ["Z1","Y2"],
		"U_Y1X2": ["Y1","X2"],
		"U_X1X2": ["X1","X2"],
		"Z1": ["X1","Z2","Y1"],
		"X1": ["Y1"],
		"Y1": ["Z2","X2"],
		"Z2": ["X2","Y2"],
		"X2": ["Y2"],
		"Y2": []
	}
	node_positions = {
		"U_Z1Y2": (-450,100),
		"U_Y1X2": (-450,0),
		"U_X1X2": (-300,0),
		"Z1": (-630,90),
		"X1": (-690,30),
		"Y1": (-570,30),
		"Z2": (-540,90),
		"X2": (-450,30),
		"Y2": (-360,30)
	}
	X = ["X1", "X2"]
	Y = ["Y1","Y2"]
	
	return [graph_dict, node_positions, X, Y]


def mSBD3():
	graph_dict = {
		"U_Z1X1": ["Z1","X1"],
		"U_Z1Z2": ["Z1","Z2"],
		"U_Z3Y": ["Z3","Y"],
		"X1": ["Z1","X2", "Y"],
		"Z1": ["X2"],
		"X2": ["Z2","Y"],
		"Z2": ["Z3"],
		"Z3": [],
		"Y": [],
	}
	node_positions = None
	X = ["X1", "X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def mSBD_minimum():
	graph_dict = {
		"U_X1Zb": ["X1","Zb"],
		"U_X1Za": ["X1","Za"],
		"U_ZaY": ["Za","Y"],
		"U_ZcY": ["Zc","Y"],
		"Zb": ["Zc", "Za"],
		"Zc": ["Za"],
		"Za": ["X2"],
		"X1": ["Zb", "Za", "X2","Y"],
		"X2": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_X1Zb": (-40,-10),
		"U_X1Za": (-38.13,-20.83),
		"U_ZaY": (-30.39,-52.15),
		"U_ZcY": (-24,-48),
		"Zb": (-35,-17),
		"Zc": (-25,-18),
		"Za": (-34,-31),
		"X1": (-45,-15),
		"X2": (-32.65,-53.73),
		"Y": (-32,-80)
	}
	# node_positions = None 
	X = ["X1", "X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]


def Napkin_FD():
	graph_dict = {
		"U_WX": ["W","X"],
		"U_WZ": ["W","Z"],
		"U_XY": ["X","Y"],
		"W": ["R"],
		"R": ["X"],
		"X": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_WX": (10,150),
		"U_WZ": (190,135),
		"U_XY": (150,-50),
		"W": (100,200),
		"R": (50,100),
		"X": (0,0),
		"Z": (200,0),
		"Y": (300,0),
	}

	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def Tian_mSBD():
	graph_dict = {
		"U_Z1X2": ["Z1","X2"],
		"U_Z1Z2": ["Z1","Z2"],
		"U_Z2Y": ["Z2","Y"],
		"Z1": ["X1"],
		"Z2": ["X2"],
		"X1": ["Y"],
		"X2": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_Z1X2": (35,45),
		"U_Z1Z2": (-25,50),
		"U_Z2Y": (70,-70),
		"Z1": (0,100),
		"Z2": (0,0),
		"X1": (100,100),
		"X2": (100,0),
		"Y": (150,50),
	}
	X = ["X1", "X2"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def FD0():
	graph_dict = {
		"U_XY": ["X", "Y"],
		"X": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		"X": (0,0),
		"Z": (100,0),
		"Y": (200,0),
		"U_XY": (100,-25)
	}
	X = ["X"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def FD1():
	graph_dict = {
		"U_CX": ["C","X"],
		"U_CY": ["C","Y"],
		"U_XY": ["X", "Y"],
		"C": ["X","Z","Y"],
		"X": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_CX": (50,50),
		"U_CY": (150,50),
		"C": (100,50),
		"X": (0,0),
		"Z": (100,0),
		"Y": (200,0),
		"U_XY": (100,-25)
	}
	X = ["X"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def FD2():
	graph_dict = {
		"U_XY": ["X", "Y"],
		"X": ["A"],
		"A": ["B", "C"],
		"B": ["D"],
		"C": ["D"],
		"D": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_XY": (100,-25),
		"X": (0,0),
		"A": (50,0),
		"B": (100,25),
		"C": (100,0),
		"D": (150,0),
		"Y": (200,0),
	}
	X = ["X"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def FD3():
	graph_dict = {
		"U_XY": ["X", "Y"],
		"X": ["A"],
		"A": ["Y"],
		"Y": [],
		"C": ["Y"],
		"D": ["C","B"],
		"B": ["Y"]
	}
	node_positions = {
		"U_XY": (100,-25),
		"X": (0,0),
		"A": (100,0),
		"Y": (200,0),
		"C": (300,0),
		"D": (400,0),
		"B": (300,25),
	}
	X = ["X"]
	Y = ["Y"]
		
	return [graph_dict, node_positions, X, Y]

def FD4():
	graph_dict = {
		"U_XY": ["X", "Y"],
		"X": ["A"],
		"A": ["B"],
		"B": ["C", "Y"],
		"C": ["Y"],
		"Y": [],
		"D": ["Y"],
		"E": ["D","Y"],
		"U_AD": ["A", "D"]
	}
	node_positions = {
		"U_XY": (100,50),
		"X": (0,0),
		"A": (100,0),
		"B": (200,0),
		"C": (300,25),
		"Y": (400,0),
		"D": (200,-50),
		"E": (300,-50),
		"U_AD": (100,-50)
	}
	X = ["X"]
	Y = ["Y"]
		
	return [graph_dict, node_positions, X, Y]

def FD5():
	graph_dict = {
		"U_XY": ["X", "Y"],
		"U_XC": ["X", "C"],
		"U_BY": ["B", "Y"],
		"X": ["A"],
		"A": ["B"],
		"B": ["C"],
		"C": ["Y"],
		"D": ["A","Y"],
		"Y": []
	}
	node_positions = {
		"U_XY": (200,75),
		"U_XC": (150,25),
		"U_BY": (300,25),
		"X": (0,0),
		"A": (100,0),
		"B": (200,0),
		"C": (300,0),
		"D": (250,-50),
		"Y": (400,0)
	}
	X = ["X"]
	Y = ["Y"]
		
	return [graph_dict, node_positions, X, Y]

def UCA1():
	graph_dict = {
		"U_CX2": ["C","X2"],
		"U_CY": ["C","Y"],
		"U_X1X2": ["X1","X2"],
		"U_X2Y": ["X2", "Y"],
		"C": ["X1","X2","Z","Y"],
		"X1": ["X2","Z","Y"],
		"X2": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_CX2": (50,50),
		"U_X1X2": (-57,7),
		"U_CY": (150,50),
		"C": (100,50),
		"X1": (-50,25),
		"X2": (0,0),
		"Z": (100,0),
		"Y": (200,0),
		"U_X2Y": (100,-25)
	}
	X = ["X1","X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def UCA2():
	graph_dict = {
		# "U_CX2": ["C","X2"],
		"U_CY": ["C","Y"],
		"U_X1X2": ["X1","X2"],
		"U_X2Y": ["X2", "Y"],
		"C": ["X1","Y"],
		"X1": ["X2","Z","Y"],
		"X2": ["Z"],
		"Z": ["Y"],
		"Y": []
	}
	node_positions = {
		# "U_CX2": (50,50),
		"U_X1X2": (-57,7),
		"U_CY": (150,50),
		"C": (100,50),
		"X1": (-50,25),
		"X2": (0,0),
		"Z": (100,0),
		"Y": (200,0),
		"U_X2Y": (100,-25)
	}
	X = ["X1","X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]
		
def Rina1():
	graph_dict = {
		"U_CX1": ["C","X1"],
		"U_X1B": ["X1","B"],
		"U_BY": ["B", "Y"],
		"C": ["A","X2"],
		"X1": ["A"],
		"A": ["B"],
		"B": ["D"],
		"X2": ["D"],
		"D": ["Y"],
		"Y": []
	}
	node_positions = None
	# node_positions = {
	# 	# "U_CX2": (50,50),
	# 	"U_X1X2": (-57,7),
	# 	"U_CY": (150,50),
	# 	"C": (100,50),
	# 	"X1": (-50,25),
	# 	"X2": (0,0),
	# 	"Z": (100,0),
	# 	"Y": (200,0),
	# 	"U_X2Y": (100,-25)
	# }
	X = ["X1","X2"]
	Y = ["Y"]
	
	return [graph_dict, node_positions, X, Y]

def unID1():
	graph_dict = {
		"U_XY": ["X","Y"],
		"U_AY": ["A", "Y"],
		"X": ["A"],
		"A": ["B"], 
		"B": ["Y"],
		"Y": []
	}
	node_positions = {
		"U_XY": (150,50),
		"U_AY": (200,-50),
		"X": (0,0),
		"A": (100,0),
		"B": (200,0),
		"Y": (300,0)
	}
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def unID2():
	graph_dict = {
		"U_XB": ["X","B"],
		"U_AY": ["A", "Y"],
		"U_BY": ["B", "Y"],
		"X": ["Y"],
		"A": ["X"], 
		"B": ["A"],
		"Y": []
	}
	node_positions = None
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def any_graph():
	graph_dict = graph_dict = {
		"V1": ["X", "V2"],
		"V4": ["X", "V2"],
		"X": [],
		"V2": ["Y"],
		"V3": ['V2', 'Y'],
		"Y": []	
	}
	node_positions = None
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def any_graph2():
	graph_dict = graph_dict = {
		"V1": ["X", "V2"],
		"X": ['V2'],
		"V2": ["V3"],
		"V3": ['Y'],
		"Y": []	
	}
	node_positions = None
	X = ["X"]
	Y = ["Y"]

	return [graph_dict, node_positions, X, Y]

def any_graph3():
	graph_dict = {'U_Y2_Y1': ['Y1', 'Y2'], 'Y1': [], 'Y2': ['X2', 'Y1', 'V1'], 'U_V1_X1': ['X1', 'V1'], 'X1': ['Y2'], 'V1': ['Y1'], 'X2': ['V1', 'Y1']}
	node_positions = None
	X = ['X1','X2']
	Y = ['Y1','Y2']
	return [graph_dict, node_positions, X, Y]






