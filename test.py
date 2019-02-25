#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2,csv
import lenet5_infernece

			
def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot

def load_mnist(path, kind='train'):
	"""Load MNIST data from `path`"""
	if kind=='train':
		labels_path=os.path.abspath('../mnist/train-labels-idx1-ubyte')		
		images_path=os.path.abspath('../mnist/train-images-idx3-ubyte')
	else:
		labels_path=os.path.abspath('../mnist/t10k-labels-idx1-ubyte')		
		images_path=os.path.abspath('../mnist/t10k-images-idx3-ubyte')
	
	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II',
								 lbpath.read(8))
		labels = np.fromfile(lbpath,
							 dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII",
											   imgpath.read(16))
		images = np.fromfile(imgpath,
							 dtype=np.uint8).reshape(len(labels), 784)

	return images, labels

BATCH_SIZE = 100 
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 5000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "./lenet5/"
MODEL_NAME = "lenet5_model"
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
display_step = 100
learning_rate_flag=True


def train(X_test,y_test_lable):
	x_ = tf.placeholder(tf.float32, [None, INPUT_NODE],name='x-input')	
	x = tf.reshape(x_, shape=[-1, 28, 28, 1])
	
	y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='y-input')
	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	cosine,loss = lenet5_infernece.inference(x,False,regularizer,tf.argmax(y_,1))

	# Evaluate model
	pred_max=tf.argmax(cosine,1)
	y_max=tf.argmax(y_,1)
	correct_pred = tf.equal(pred_max,y_max)
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
	# 初始化TensorFlow持久化類。
	saver = tf.train.Saver()
	with tf.Session() as sess:
	        saver.restore(sess,"lenet5/lenet5_model") 
		X_test = np.reshape(X_test, (
                                        X_test.shape[0],
                                        IMAGE_SIZE,
                                        IMAGE_SIZE,
                                        NUM_CHANNELS))

		acc = sess.run(accuracy, feed_dict={x: X_test, y_:y_test_lable})
                print('Test accuracy: %.2f%%' % (acc * 100))
		
def main(argv=None):
	#mnist = input_data.read_data_sets("./mnist", one_hot=True)
	X_test, y_test = load_mnist('mnist', kind='t10k')					 #X_test=10000x784
	mms=MinMaxScaler()
        X_test=mms.fit_transform(X_test) 
        y_test_lable = encode_labels(y_test,10) 
	train(X_test,y_test_lable)

if __name__ == '__main__':
        start = time.time()
	main()
        end = time.time()
        print  end-start
        print  'I have trained %d mins and %d seconds'%((end-start)/60,(end-start)%60)
