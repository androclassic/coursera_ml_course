import utils as lr_utils
import shallow_nn.shallow_nn as snn
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from PIL import Image

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


# Build a model with a n_h-dimensional hidden layer
parameters = snn.nn_model(train_set_x, train_set_y, n_h = 4, num_iterations = 3000, print_cost=True, learning_rate = 0.02)


# Predict test/train set examples 
Y_prediction_test = snn.predict(parameters, test_set_x)
Y_prediction_train = snn.predict(parameters, train_set_x)


# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))

