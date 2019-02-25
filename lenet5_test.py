#coding:utf-8
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import lenet5_infernece
import lenet5_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
from matplotlib import pyplot as plt

img_num=[0]*20

def evaluate(X_test,y_test_lable,My_Yd):
	with tf.Graph().as_default() as g:
		# 定義輸出為4維矩陣的placeholder
		x_ = tf.placeholder(tf.float32, [None, lenet5_train.INPUT_NODE],name='x-input')	
		x = tf.reshape(x_, shape=[-1, 28, 28, 1])
	
		y_ = tf.placeholder(tf.float32, [None, lenet5_train.OUTPUT_NODE], name='y-input')
	
		regularizer = tf.contrib.layers.l2_regularizer(lenet5_train.REGULARIZATION_RATE)
		cosine,loss= lenet5_infernece.inference(x,False,regularizer,tf.argmax(y_,1))
		global_step = tf.Variable(0, trainable=False)

		# Evaluate model
		pred_max=tf.argmax(cosine,1)
		y_max=tf.argmax(y_,1)
		correct_pred = tf.equal(pred_max,y_max)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
		batchsize=20
		test_batch_len =int( X_test.shape[0]/batchsize)
		test_acc=[]
		
		test_xs = np.reshape(X_test, (
					X_test.shape[0],
					lenet5_train.IMAGE_SIZE,
					lenet5_train.IMAGE_SIZE,
					lenet5_train.NUM_CHANNELS))
		
		# 'Saver' op to save and restore all the variables
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess,"lenet5/lenet5_model")
	        	My_test_pred=sess.run([pred_max], feed_dict={x: test_xs[:20]})
                        print a
			print("期望值：",My_Yd)
			print("預測值：",My_test_pred)
			My_acc = sess.run(accuracy, feed_dict={x: test_xs, y_: y_test_lable})
			print('Test accuracy: %.2f%%' % (My_acc * 100))
			return
def main(argv=None):
	#### Loading the data   #自己手寫的20個數字
	My_X =np.zeros((20,784), dtype=int) 
	#自己手寫的20個數字對應的期望數字
	My_Yd =np.array([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9], dtype=int) 

	#輸入20個手寫數字圖檔28x28=784 pixel，
	Input_Numer=[0]*20
	Input_Numer[0]="ziji/m0.png"
	Input_Numer[1]="ziji/m1.png"
	Input_Numer[2]="ziji/m2.png"
	Input_Numer[3]="ziji/m3.png"
	Input_Numer[4]="ziji/m4.png"
	Input_Numer[5]="ziji/m5.png"
	Input_Numer[6]="ziji/m6.png"
	Input_Numer[7]="ziji/m7.png"
	Input_Numer[8]="ziji/m8.png"
	Input_Numer[9]="ziji/m9.png"
	Input_Numer[10]="ziji/n0.png"
	Input_Numer[11]="ziji/n1.png"
	Input_Numer[12]="ziji/n2.png"
	Input_Numer[13]="ziji/n3.png"
	Input_Numer[14]="ziji/n4.png"
	Input_Numer[15]="ziji/n5.png"
	Input_Numer[16]="ziji/n6.png"
	Input_Numer[17]="ziji/n7.png"
	Input_Numer[18]="ziji/n8.png"
	Input_Numer[19]="ziji/n9.png"
	mms=MinMaxScaler()
	for i in range(20):	 #read 20 digits picture
		img = cv2.imread(Input_Numer[i],0)	  #Gray
                img=cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
		#'''
                for x in range(0, img.shape[0]):
  		    for y in range(0, img.shape[1]):
        		if img[x][y] >= 105:
            		     img[x][y] = 0
   			else:
           	             img[x][y] = 255
                #'''
		img_num[i]=img.copy()
		img=img.reshape(My_X.shape[1])
		My_X[i] =img.copy()
        print 'resize hou\n',My_X[0],'\n',My_X[0].shape
	My_test=mms.fit_transform(My_X)
        print 'aifer mms\n',My_test[0],'\n',My_test[0].shape
	My_label_ohe = lenet5_train.encode_labels(My_Yd,10)
	##============================
	
	evaluate(My_test,My_label_ohe,My_Yd)

if __name__ == '__main__':
	main()
