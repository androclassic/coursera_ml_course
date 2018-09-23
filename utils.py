import cv2
import numpy as np
import random
import math

def add_image(data, img, y):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	c1,c2,c3 = cv2.split(hsv)
	img = [c1,c2 ]
	data.append([img, y])

def load_data():
	data = []
	for x in range(350):
		img = cv2.imread('data/sensors_on/pattern_im_{0}.png'.format(x))
		if(img is not None):
			add_image(data, img, 0)
			add_image(data, cv2.flip( img, 1 ), 0)

	for x in range(350):
		img = cv2.imread('data/no_sensors/pattern_im_{0}.png'.format(x))
		if(img is not None):
			add_image(data, img, 0)
			add_image(data, cv2.flip( img, 1 ), 0)

	for x in range(350):
		img = cv2.imread('data/sensors_occluded/pattern_im_{0}.png'.format(x))
		if(img is not None):
			add_image(data, img, 1)
			add_image(data, cv2.flip( img, 1 ), 1)

	random.shuffle(data)
	random.shuffle(data)
	random.shuffle(data)

	split_idx = int(len(data) * 0.8)

	train = data[0:split_idx]
	test = data[split_idx:-1]


	X = np.array([item[0] for item in train])
	Y = np.array([item[1] for item in train])
	X_t = np.array([item[0] for item in test])
	Y_t = np.array([item[1] for item in test])


	return X,Y.reshape(1,-1),X_t,Y_t.reshape(1,-1), ['not_occluded', 'occluded']



def random_mini_batches(X, Y, mini_batch_size = 64):
    """
    Creates a list of random minibatches from (X, Y)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    m = X.shape[1]                  # number of training examples
    mini_batches = []
        
    #shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((1,m))

    # partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) 
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : (k+1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : ]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def load_batch_cifar(filename):
	import pickle
	f = open(filename, 'rb')
	print('loading', filename)
	dict = pickle.load(f,  encoding='bytes')
	images = dict[b'data']
	images = np.reshape(images, (10000, 3, 32, 32))
	labels = dict[b'labels']
	imagearray = np.array(images)   #   (10000, 3072)
	labelarray = np.array(labels)   #   (10000,)
	return imagearray, labelarray

def selected_class_data(x,y, idx):
	yes_idx = np.where( y == idx)
	no_idx = np.where( y != idx)
	sel_x = x[ yes_idx]  
	sel_x_no = x[no_idx]  

	sel_y = np.ones(len(sel_x)).astype(int)
	sel_y_no = np.zeros(len(sel_x_no)).astype(int)   
	max_len = np.min([len(sel_y), len(sel_y_no)])

	data_x =np.append(sel_x, sel_x_no[:max_len], 0)
	data_y= np.append(sel_y, sel_y_no[:max_len], 0)

	s = np.arange(len(data_y))
	np.random.shuffle(s)

	return data_x[s], data_y[s]



def load_data_cifar(binary_data_class_idx = None):
	import pickle
	path = 'data/cifar-10-batches-py/'
	train_file = 'data_batch_'
	test_file = 'test_batch'
	meta_file = 'batches.meta'

	f = open(path+meta_file, 'rb')
	dict = pickle.load(f,  encoding='bytes')

	train_x = np.array([])
	train_y = np.array([])

	for i in range(5):
		x, y = load_batch_cifar(path+train_file+str(i+1))
		if(train_x.size == 0):
			train_x  = x
		else:
			train_x =np.append(x, train_x, 0)
		if(train_y.size == 0):
			train_y = y
		else:
			train_y= np.append(y, train_y, 0)


	test_x, test_y = load_batch_cifar(path+test_file)

	labels_names = dict[b'label_names']

	if(binary_data_class_idx != None):
		train_x, train_y = selected_class_data(train_x, train_y, binary_data_class_idx)
		test_x, test_y = selected_class_data(test_x, test_y , binary_data_class_idx)
		labels_names = [b'not_'+labels_names[binary_data_class_idx],labels_names[binary_data_class_idx]]

	    
	return train_x, train_y.reshape(1,-1), test_x, test_y.reshape(1,-1), labels_names

def save_dictionary(data, filename):
	try:
		import cPickle as pickle
	except ImportError:  # python 3.x
		import pickle

	with open(filename, 'wb') as fp:
		pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_dictionary(filename):
	try:
		import cPickle as pickle
	except ImportError:  # python 3.x
		import pickle
	data = None
	with open(filename, 'rb') as fp:
		data = pickle.load(fp)
	return data