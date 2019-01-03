#students performances
from nn.modeling import *
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


def data_treat():
	data = pd.read_csv("inputs/StudentsPerformance.csv")
	table = data

	parents = pd.get_dummies(table["parental level of education"])
	
	table['parents 1'] = parents["some high school"]
	table['parents 2'] = parents["high school"]
	table['parents 3'] = parents["master's degree"]
	table['parents 4'] = parents["bachelor's degree"]
	table['parents 5'] = parents["some college"]
	table['parents 6'] = parents["associate's degree"]
	
	eth = pd.get_dummies(table["race/ethnicity"])

	table['eth 1'] = eth["group A"]
	table['eth 2'] = eth["group B"]
	table['eth 3'] = eth["group C"]
	table['eth 4'] = eth["group D"]
	table['eth 5'] = eth["group E"]
		
	table = table.replace(["female", "standard", "none"], 0)
	table = table.replace(["male", "free/reduced", "completed"], 1)

	col = table.loc[:, "math score":"writing score"]
	table["avg score"] = col.mean(axis=1)

	#table.drop("writing score", axis=1, inplace=True)
	table.drop("math score", axis =1 , inplace= True)
	table.drop("reading score", axis= 1, inplace=True)
	table.drop("parental level of education", axis=1, inplace=True)
	table.drop("race/ethnicity", axis=1, inplace=True)


	X = table.loc[:,"gender":"eth 5"]
	Y = table.loc[:,"avg score"]

	X.drop("writing score", axis=1, inplace=True)
	
	Y = (Y - min(Y))/ (max(Y) - min(Y))
	
	# X shape (#num data points, #features)
	X = X.T

	# reshape Y to become of shape(# num outputs= 1, # num data points)
	Y = Y.values
	Y = Y.reshape(1, 1000)

	'''
	Y[Y<0.9] = 0
	Y[Y> 0] = 1
	'''
	return X, Y


def script():
	X, Y = data_treat()
	print(X)
	layers_dims = [14, 64, 16,1]

	model(X,Y, layers_dims,learning_rate= 0.02, num_iterations=20000, print_cost=True)


if __name__ == "__main__":
	script()
	print("good job!")
