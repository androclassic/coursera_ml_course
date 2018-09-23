import utils as lr_utils
import logistic_regresion.logistic_regresion as lr
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
import cv2
from scipy import ndimage
from PIL import Image

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = lr_utils.load_data_cifar(3)
index = 1

print ("y = " ,(train_set_y[0,index]) , ", it's a '" , classes[train_set_y[0,index]] ,  "' picture.")
img = train_set_x_orig[index]
print(img.shape)
temp = cv2.merge((img[0,:,:],img[1,:,:],img[2,:,:]))
plt.imshow(temp)
plt.show()



m_train =  train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_py = train_set_x_orig.shape[2]
num_px = train_set_x_orig.shape[3]
num_c = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: w = " + str(num_px)+' h=' + str(num_py))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_py) + ", " + str(num_c)+')')
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

d = lr.model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 3000, learning_rate = 0.042, print_cost = True)

    
# Predict test/train set examples 
Y_prediction_test = lr.predict(d['w'], d['b'], test_set_x)
Y_prediction_train = lr.predict(d['w'], d['b'], train_set_x)


# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_set_y)) * 100))




w_img = d["w"]
np.savetxt('test.txt',w_img)
temp = []
#temp = np.append(w_img[:,0], np.zeros(num_px * num_py))
temp = w_img[:,0]
temp = temp.reshape((num_c,num_py,num_px)) * 255
temp = cv2.merge((temp[0,:,:],temp[1,:,:],temp[2,:,:]))
plt.imshow(temp)
plt.show()


# Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()