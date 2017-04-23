#Define network and its trainer

import tensorflow as tf
import numpy as np

class cnn:

	def __init__(self, nFeatures, nLabels, nLayers = 3):

		
		self.nLabels = nLabels
		self.nLayers = nLayers
		self.nFeatures = nFeatures	

	@staticmethod 
	def init_weights(shape):
		weights = tf.truncated_normal(shape,stddev = 0.1)
		return tf.Variable(weights)

	@staticmethod
	def init_biases(shape):
		biases = tf.zeros(shape)
		return tf.Variable(biases)

	@staticmethod
	def conv2d(inputs,f):
		return tf.nn.conv2d(inputs,f,strides = [1,1,1,1,],padding = "SAME")

	@staticmethod
	def max_pool_2x2(inputs):
		return tf.nn.max_pool(inputs,ksize = [1,2,2,1],strides = [1,2,2,1,],padding = "SAME")

	@staticmethod
	def conv2d(inputs,f):
		return tf.nn.conv2d(inputs,f,strides = [1,1,1,1,],padding = "SAME")

	@staticmethod
	def relu(inputs):
		return tf.nn.relu(inputs)

	@staticmethod
	def dropout(inputs,keep_prob):
		return tf.nn.dropout(inputs,keep_prob)

	def create_conv_net(self,inputs , keep_prob, nLabels, nLayers):

		inputs = tf.reshape(inputs,[-1,28,28,1])

		nFeature_maps = 16
		size_init = inputs.get_shape()[2].value
		size = size_init
		W = []
		B = []

		for layer in range(nLayers):

			w = cnn.init_weights([3,3,inputs.get_shape()[3].value,nFeature_maps])
			b = cnn.init_biases([nFeature_maps])
			conv = cnn.dropout(cnn.relu(cnn.conv2d(inputs,w)+b),keep_prob)
			pooled = cnn.max_pool_2x2(conv)

			nFeature_maps *= 2
			inputs = pooled

			size = size/2
			W.append(w)
			B.append(b)

		inputs = conv #Ignore last max pool layer
		size = size*2
		offset = size_init - size

		#Add two fully connected layers
		flat_size = int(size*size*inputs.get_shape()[3].value) 
		w = cnn.init_weights([flat_size,1000])
		b = cnn.init_biases([1000])
		flat = tf.reshape(inputs, [-1,flat_size])
		fcl1 = cnn.dropout(cnn.relu(tf.matmul(flat,w) + b),keep_prob)
		W.append(w)
		B.append(b)

		w = cnn.init_weights([1000,nLabels])
		b = cnn.init_biases([nLabels])
		fcl2 = cnn.dropout(tf.matmul(fcl1,w) + b,keep_prob)
		W.append(w)
		B.append(b)

		return fcl2,W,B


	def predict(self,x_, keep_prob):

		self.logits,self.W,self.B = self.create_conv_net(x_, keep_prob,nLabels = self.nLabels, nLayers = self.nLayers)

		predictions = tf.nn.softmax(self.logits)
		return predictions




				



