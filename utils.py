import cv2
import numpy as np
import random

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