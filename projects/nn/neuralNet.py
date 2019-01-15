import numpy as np

from nn.utils import sigmoid, tanh, relu, sigmoid_back, tanh_back, relu_back

def init_params(layer_dims): # e.g layer_dims = [128, 64, 1]
    # create param dict
    params = {}
    # create threshold value of num layer in network for loop
    L = len(layer_dims)
    # loop over each layer and init rand vals for W and zeros for b
    for l in range(1,L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))

        assert(params['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(params['b' + str(l)].shape == (layer_dims[l], 1))


    return params

def forward(A,W, b):
    # multiply values by weights and add bias
    Z = np.dot(W,A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    # return computed value of Z and original inputs
    return Z, cache

def activation_forward(A_prev, W, b, activation):
    # use the forward function then the activation func
    if activation == "sigmoid":
        Z, linear_cache = forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        Z, linear_cache = forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    elif activation == "tanh":
        Z, linear_cache = forward(A_prev,W,b)
        A, activation_cache = tanh(Z)

    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    # return new value of cell and original vules
    return A, cache

def model_forward(X, params):
    caches = []
    A = X
    # number of layers in neural net
    L = len(params)//2
    # on each layer use relu than tanh
    for l in range(1, L):
        A_prev = A

        A, cache = activation_forward(A_prev, params['W' + str(l)],
                        params['b' + str(l)], activation= "relu")

        caches.append(cache)

    AL, cache = activation_forward(A, params['W' + str(L)],
                        params['b' + str(L)], activation ="sigmoid")

    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches


def res_forward(A,W,b):
    # uses relu and resnet technique
    short_cut = A
    
    Z = np.dot(W,A) + b
    linear_cache = (A, W, b)
    A, activation_cache =  relu(Z) + np.dot(W, short_cut)

    cache = (linear_cache, activation_cache)
    return A, cache

def res_model_forward(X, params):
    caches = []
    A = X
    # number of layers in neural net
    L = len(params)//2
    # on each layer use relu than tanh
    for l in range(1, L):
        A_prev = A

        A, cache = res_forward(A_prev, params['W' + str(l)], params['b' + str(l)])

        caches.append(cache)

    AL, cache = activation_forward(A, params['W' + str(L)],
                        params['b' + str(L)], activation ="sigmoid")

    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

def getCost(AL, Y):
    m = Y.shape[1]
    # compare final output with Y using Log Classification cost function
    cost_log = (-1 / m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))
    # compare using MSE regression cost function
    mse = (np.square(AL - Y)).mean()
    cost = np.squeeze(mse)

    assert(cost.shape ==())

    return cost

def backward(dZ, cache):
    # decompose variables to go backwards
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, cache[0].T) / m
    db = np.sum(dZ, axis=1, keepdims= True) /m
    dA_prev = np.dot(cache[1].T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev , dW, db

def activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    # choose activation func and use deriv this func of activ on
    # the derivative of Z
    if activation == "relu":
        dZ = relu_back(dA, activation_cache)
        # retrieve differente values from backward func
        dA_prev, dW, db = backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_back(dA, activation_cache)
        # retrieve differente values from backward func
        dA_prev, dW, db = backward(dZ, linear_cache)

    elif activation == "tanh":
        dZ = tanh_back(dA, activation_cache)
        # retrieve differente values from backward func
        dA_prev, dW, db = backward(dZ, linear_cache)
    

    return dA_prev, dW, db



def model_backward(AL, Y, caches):

    grads = {}
    # number layers
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    # compute derivative of activation layer
    dAL = -(np.divide(Y,AL) - np.divide(1 - Y, 1-AL))

    current_cache = caches[L-1]
    # get gradients of last activation layer
    grads["dA" + str(L)], grads["dW"+str(L)], grads["db"+ str(L)] = activation_backward(dAL, current_cache,'sigmoid')
    
    # get grads for all other layers
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 2)],
            current_cache, 'relu')
        grads['dA'+str(l+1)] = dA_prev_temp
        grads['dW'+str(l+1)] = dW_temp
        grads['db'+str(l+1)] = db_temp

    return grads

def update_params(params, grads, learning_rate):
    L = len(params)//2
    # update parameters of weight and bias with learning rate
    for l in range(1, L+1, 1):
        params["W" +str(l)] = params["W" +str(l)] - learning_rate * grads["dW"+str(l)]
        params["b" +str(l)] = params["b" +str(l)] - learning_rate * grads["db"+str(l)]

    return params
