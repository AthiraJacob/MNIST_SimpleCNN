'''
Simple CNN model for MNIST data
Author: Athira
'''

import numpy as np
import tensorflow as tf
import argparse 
from network import cnn
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
import input_data

data_dir = '/cis/home/ajacob/Documents/enlitic/data/'

FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--nLayers', type = int,default = 3, help = 'Number of conv layers of CNN')
parser.add_argument('--nEpochs',type = int, default = 100, help = 'Number of epochs to train for')
FLAGS, unparsed = parser.parse_known_args()

mnist = input_data.read_data_sets(data_dir, one_hot=True)

sample = mnist.train.next_batch(1)

nFeatures = sample[0].size
nLabels = sample[1].size

x_ = tf.placeholder(tf.float32,shape = [None, 784])
y_true = tf.placeholder(tf.float32,shape = [None,10])
keep_prob = tf.placeholder(tf.float32)

net = cnn(nFeatures,nLabels,FLAGS.nLayers)
pred = net.predict(x_,keep_prob)
loss = tf.losses.softmax_cross_entropy(onehot_labels = y_true,logits = pred)
trainer = tf.train.AdamOptimizer().minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(pred,1),tf.arg_max(y_true,1)),tf.float32))

epoch = 0
batchSize_train = 700
batchSize_test = 500

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

train_acc = []
test_acc = []

while(epoch<FLAGS.nEpochs):

	train_imgs,train_labels = mnist.train.next_batch(batchSize_train)
	trainer.run(feed_dict = {x_ : train_imgs,y_true : train_labels,keep_prob : 0.5})
	acc_train = accuracy.eval(feed_dict = {x_ : train_imgs,y_true : train_labels,keep_prob: 1.0})
	print('Epoch ' + str(epoch) + ' Training accuracy: '+ str(acc_train))
	train_acc.append(acc_train)
	
	#Verification
	test_imgs,test_labels = mnist.train.next_batch(batchSize_test)
	acc_test = accuracy.eval(feed_dict = {x_ : test_imgs,y_true : test_labels,keep_prob: 1.0})
	print(' Verification accuracy: '+ str(acc_test))
	test_acc.append(acc_test)

	if epoch==0:
		W0 = net.W
		B0 = net.B

	if epoch == FLAGS.nEpochs-1:
		W_final = net.W
		B_final = net.B

	epoch += 1



t = np.arange(FLAGS.nEpochs)
plt.plot(t,train_acc,'r',t,test_acc,'b')
plt.savefig('graph')

# for i in range(len(W_final)):
# 	plt.hist(W_final[i].eval().reshape((-1)),normed=True)
# 	plt.savefig('hist_new'+str(i))
# 	plt.close()

# for i in range(len(W0)):
# 	plt.hist(W0[i].eval().reshape((-1)),normed=True)
# 	plt.savefig('hist_old'+str(i))
# 	plt.close()