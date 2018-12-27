#students performances
from nn.modeling import *
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


def data_treat():
	data = pd.read_csv('inputs/StudentsPerformance.csv')
	table = data
		
	table.drop("race/ethnicity", axis= 1, inplace=True)
	table.drop("parental level of education", axis=1, inplace=True)
		
	table = table.replace(["female", "standard", "none"], 0)
	table = table.replace(["male", "free/reduced", "completed"], 1)

	col = table.loc[:, "math score":"writing score"]
	table['avg score'] = col.mean(axis=1)

	table.drop("writing score", axis=1, inplace=True)

	X = table.loc[:,"gender":"reading score"]
	Y = table.loc[:,"avg score"]

	X['math score'] = (X['math score'] - min(X['math score'])) / (max(X['math score']) - min(X['math score']))
	X['reading score'] = (X['reading score'] - min(X['reading score'])) / (max(X['reading score']) - min(X['reading score']))

	Y = (Y - min(Y))/ (max(Y) - min(Y))
	
	# X shape (#features, #num data points)
	X = X.T

	# reshape Y to become of shape(# num outputs= 1, # num data points)
	Y = Y.values
	Y = Y.reshape(1, 1000)

	Y[Y<Y.mean()] = 0
	Y[Y> 0.5] = 1

	print(X)
	return X, Y


def script():
	X, Y = data_treat()
	layers_dims = [5, 10, 5 ,1]

	model(X,Y, layers_dims,learning_rate= 0.08, num_iterations=18700, print_cost=True)


if __name__ == "__main__":
	script()
	print("good job!")