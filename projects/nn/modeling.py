# implement model
from nn.neuralNet import init_params, forward, activation_forward, model_forward
from nn.neuralNet import getCost, backward, activation_backward, model_backward, update_params
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def plot_it(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

def model(X, Y, layers_dims, learning_rate=0.001, num_iterations=10000, print_cost=False, RESNET = False):
    costs = []

    parameters = init_params(layers_dims)
    
    for i in range(0, num_iterations):
        if RESNET:
            AL, caches = res_model_forward(X, parameters)
        else:
            AL, caches = model_forward(X, parameters)
        
        cost = getCost(AL, Y)
        
        grads = model_backward(AL, Y, caches)
        
        parameters = update_params(parameters, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    
    plot_it(costs, learning_rate)
    
    return parameters
