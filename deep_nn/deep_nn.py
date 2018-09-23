# Package imports
import numpy as np
import utils as lr_utils

def sigmoid(z):
    s =  1/ (1 + np.exp(-1 * z))
    return s,z

#compute dZ[l] = dA[l] * g'(Z[l])      
def sigmoid_backward(dA, activation_cache):
	simoid_z,z = sigmoid(activation_cache)
	dZ = dA  * ( simoid_z * (1 - simoid_z))
	return dZ


def relu(z):
	s = np.maximum(0,z)
	return s,z

#compute dZ[l] = dA[l] * g'(Z[l])      
def relu_backward(dA, activation_cache):
	return (activation_cache >= 0) * dA


 #initialize_parameters
def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    """
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] =  np.random.randn(layer_dims[l],layer_dims[l-1]) *  np.sqrt(2./layer_dims[l-1])
        parameters['b' + str(l)] = np.zeros((layer_dims[l],1))
                
    return parameters

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
    Arguments:
    parameters -- python dictionary containing your parameters.
    Returns:
    v -- python dictionary containing the current velocity.
    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    
    # Initialize velocity
    for l in range(L):
        v["dW" + str(l+1)] = np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] =  np.zeros(parameters["b" + str(l+1)].shape)
        
    return v

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL" 
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    
    Arguments:
    parameters -- python dictionary containing your parameters.
    Returns: 
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.

    """
    
    L = len(parameters) // 2 # number of layers in the neural networks
    v = {}
    s = {}
    
    # Initialize v, s. Input: "parameters". Outputs: "v, s".
    for l in range(L):
        v["dW" + str(l+1)] =  np.zeros(parameters["W" + str(l+1)].shape)
        v["db" + str(l+1)] =  np.zeros(parameters["b" + str(l+1)].shape)
        s["dW" + str(l+1)] =  np.zeros(parameters["W" + str(l+1)].shape)
        s["db" + str(l+1)] =  np.zeros(parameters["b" + str(l+1)].shape)
    
    return v, s

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    """
    
    Z = W.dot(A) + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

# linear_activation_forward
def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    """
    
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        ### START CODE HERE ### (≈ 2 lines of code)
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
        ### END CODE HERE ###
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache

# GRADED FUNCTION: L_model_forward

def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2                  # number of layers in the neural network
    
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)] , parameters['b' + str(l)] , "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)] , parameters['b' + str(L)] , "sigmoid")
    caches.append(cache)
    
    assert(AL.shape == (1,X.shape[1]))
            
    return AL, caches

#compute_cost
def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).
    Returns:
    cost -- cross-entropy cost
    """
    m = Y.shape[1]

    # Compute loss from aL and y.
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)
    cost = -np.sum(logprobs)/m
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    return cost

def compute_cost_with_regularization(AL, Y, parameters, lambd):
	"""
	Implement the cost function with L2 regularization. See formula (2) above.
	"""
	m = Y.shape[1]

	cross_entropy_cost = compute_cost(AL, Y) # This gives you the cross-entropy part of the cost
	L2_regularization_cost = 0
	L = len(parameters) // 2 # number of layers in the neural network
	for l in range(L):
		W = parameters["W" + str(l+1)] 
		L2_regularization_cost += np.sum(np.square(W))

	L2_regularization_cost *=  lambd/(2*m)
	cost = cross_entropy_cost + L2_regularization_cost
	return cost


#linear_backward
def linear_backward(dZ, cache, lambd):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = dZ.dot(A_prev.T)/m + (lambd*W)/m
    db = np.sum(dZ, axis = 1, keepdims = True) / m
    dA_prev = W.T.dot(dZ)
    
    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)
    
    return dA_prev, dW, db

 #linear_activation_backward
def linear_activation_backward(dA, cache, activation, lambd):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    """
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    
    return dA_prev, dW, db

# GRADED FUNCTION: L_model_backward

def L_model_backward(AL, Y, caches, lambd):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector 
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL)) # derivative of cost with respect to AL
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
    ### START CODE HERE ### (approx. 2 lines)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        # Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp =  linear_activation_backward(grads["dA" + str(l + 1)], current_cache, "relu", lambd)
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters_with_gd(parameters, grads, learning_rate):

    L = len(parameters) // 2 # number of layers in the neural networks
    
    for l in range(L):
        
        # update parameters
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] - learning_rate * grads['dW' + str(l+1)]
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)] - learning_rate  * grads['db' + str(l+1)]
        
    return parameters


#update_parameters_with_momentum

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum
    
    Arguments:
    parameters -- python dictionary containing your parameters:
    grads -- python dictionary containing your gradients for each parameters:
    v -- python dictionary containing the current velocity:
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2 # number of layers in the neural networks
    
    # Momentum update for each parameter
    for l in range(L):
        
        # compute velocities
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] =  beta * v["db" + str(l+1)] + (1-beta) * grads['db' + str(l+1)]
        # update parameters
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] - learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)] - learning_rate * v["db" + str(l+1)]
        
    return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
    grads -- python dictionary containing your gradients for each parameters:
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
        # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
        v["dW" + str(l+1)] =  beta1 * v["dW" + str(l+1)] + (1-beta1) * grads['dW' + str(l+1)]
        v["db" + str(l+1)] =  beta1 * v["db" + str(l+1)] + (1-beta1) * grads['db' + str(l+1)]
 
        # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
        v_corrected["dW" + str(l+1)] = v["dW" + str(l+1)] / np.power(1-beta1, l)
        v_corrected["db" + str(l+1)] = v["db" + str(l+1)] / np.power(1-beta1, l)

        # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
        s["dW" + str(l+1)] =  beta2 * s["dW" + str(l+1)] + (1-beta2) * np.power(grads['dW' + str(l+1)],2)
        s["db" + str(l+1)] =  beta2 * s["db" + str(l+1)] + (1-beta2) * np.power(grads['db' + str(l+1)],2)

        # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
        s_corrected["dW" + str(l+1)] = s["dW" + str(l+1)] / np.power(1-beta2, l)
        s_corrected["db" + str(l+1)] = s["db" + str(l+1)] / np.power(1-beta2, l)

        # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
        parameters["W" + str(l+1)] =  parameters["W" + str(l+1)] -  learning_rate * (v_corrected["dW" + str(l+1)]/np.sqrt(s_corrected["dW" + str(l+1)] + epsilon))
        parameters["b" + str(l+1)] =  parameters["b" + str(l+1)] -  learning_rate * (v_corrected["db" + str(l+1)]/np.sqrt(s_corrected["db" + str(l+1)] + epsilon))

    return parameters, v, s

def save_parameters(parameters):
	lr_utils.save_dictionary(parameters, 'nn_test_params')


#L_layer_model

def L_layer_model(X, Y, layers_dims, optimizer, learning_rate = 0.0075,
	mini_batch_size = 64, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000,
	 print_cost=False, preloaded_weights = None, lambd = 0.7):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates 
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- if True, it prints the cost every 100 steps
    preloaded_weights = pretraind model

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    costs vector
    """
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    if preloaded_weights:
        for l in range(1,len(layers_dims)):
            assert(parameters["W" + str(l)].shape == preloaded_weights["W" + str(l)].shape)
        parameters = preloaded_weights
        print('weights preloaded')

    print('Using :',optimizer)
     # Initialize the optimizer
    if optimizer == "gd":
        pass # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
             
        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        minibatches = lr_utils.random_mini_batches(X, Y, mini_batch_size)

        for minibatch in minibatches:
    
            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch
            # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
            AL, caches = L_model_forward(minibatch_X, parameters)
            # Compute cost.
            cost =  compute_cost_with_regularization(AL, minibatch_Y, parameters, lambd)
            # Backward propagation.
            grads = L_model_backward(AL, minibatch_Y, caches, lambd)
            # Update parameters.
                    # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, learning_rate, beta1, beta2,  epsilon)

        # Print the cost every 100 training example
        if print_cost:
           print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
           save_parameters(parameters)
           costs.append(cost)
           
    return parameters, costs