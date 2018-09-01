import utils as lr_utils
import deep_nn.deep_nn as dnn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from PIL import Image

def predict(parameters, X):
    """
    Using the learned parameters, predicts a class for each example in X
    Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.
    """

    AL, caches = dnn.L_model_forward(X, parameters)
    predictions = AL >= 0.5
    
    return predictions



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_data()

# Example of a picture
index = 25
img = train_set_x_orig[index]
print(img.shape)
#plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[0,index]) + ", it's a '" + classes[train_set_y[0,index]] +  "' picture.")
#plt.show()


m_train =  train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_py = train_set_x_orig.shape[2]
num_px = train_set_x_orig.shape[3]
num_c = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: w = " + str(num_px)+' h=' + str(num_py))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", " + str(num_c))
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T


print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#normalize values
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


n_x = train_set_x.shape[0] # size of input layer
n_y = train_set_y.shape[0] # size of output layer
layers_dims =  (n_x, 16, 3, n_y)

learning_rate = 0.075

# Build a model with a n_h-dimensional hidden layer
parameters, costs = dnn.L_layer_model(train_set_x, train_set_y, layers_dims,  learning_rate, num_iterations = 2500, print_cost=True)

# plot the cost
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(learning_rate))



# Predict test/train set examples 
Y_prediction_test = predict(parameters, test_set_x)
Y_prediction_train = predict(parameters, train_set_x)


# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

plt.show()
