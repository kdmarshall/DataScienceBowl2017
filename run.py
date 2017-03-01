import os
import numpy as np
import pandas as pd
import dicom
from PIL import Image
import tensorflow as tf
import cv2
import math

from utils import *

tf.app.flags.DEFINE_string("stage1",'','Full training examples set.')
tf.app.flags.DEFINE_string("samples",'','Sample training examples set.')
tf.app.flags.DEFINE_string("labels",'','Full training labels set.')

# Globals
FLAGS = tf.app.flags.FLAGS
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_DEPTH = 120
HM_SLICES = 20
# HP
N_CLASSES = 2
LEARNING_RATE = 1e-3

# TF Placeholders
# [batch_size,depth,height,width,channel]
samples = tf.placeholder(tf.float32,[None, IMAGE_DEPTH, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
labels = tf.placeholder(tf.float32,[None, N_CLASSES])

def get_sample_data(examples_path,labels_path):
	# Not yet working!
	# Still playing with data reformating
	patients = [fname for fname in os.listdir(examples_path) if fname != '.DS_Store']
	labels_df = pd.read_csv(labels_path, index_col='id')
	for patient in patients:
		try:
			label = labels_df.get_value(str(patient), 'cancer')
		except:
			print("Patient {} not found".format(patient))
			continue
		file_path = os.path.join(examples_path,patient)
		slices = [dicom.read_file(os.path.join(file_path,fname))
					for fname in os.listdir(file_path)
						if fname != '.DS_Store']
		slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
		slices = [cv2.resize(np.array(each_slice.pixel_array),(IMAGE_WIDTH,IMAGE_HEIGHT))
					for each_slice in slices]
		new_slices = []
		chunk_sizes = int(math.ceil(len(slices) / HM_SLICES))
		print(chunk_sizes)
		for slice_chunk in chunks(slices,chunk_sizes):
			slice_chunk = list(map(mean, zip(*slice_chunk)))
			new_slices.append(slice_chunk)

		print(len(slices), len(new_slices))
		break
	

def conv3d(x, W):
	return tf.nn.conv3d(x,
						W,
						strides=[1, 1, 1, 1, 1],
						padding='SAME')

def maxpool3d(x):
	return tf.nn.max_pool3d(x,
							ksize=[1, 2, 2, 2, 1],
							strides=[1, 2, 2, 2, 1],
							padding='SAME',
							name=None)

def _variable_on_cpu(name, shape, initializer):
	"""
	Helper to create a Variable stored on CPU memory.
	
	Args:
		name: name of the variable
		shape: list of ints
		initializer: initializer for Variable
	Returns:
		Variable Tensor
	"""
	with tf.device('/cpu:0'):
		# dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
		dtype = tf.float32
		var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
		# var = tf.get_variable(name, shape, initializer=initializer)
	return var

def _activation_summary(x):
	"""
	Helper to create summaries for activations.
	Creates a summary that provides a histogram of activations.
	Creates a summary that measures the sparsity of activations.
	
	Args:
		x: Tensor
	Returns:
		None
	"""
	# Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
	# session. This helps the clarity of presentation on tensorboard.
	tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
	tf.contrib.deprecated.histogram_summary(tensor_name + '/activations', x)
	tf.contrib.deprecated.scalar_summary(tensor_name + '/sparsity',
	                                   tf.nn.zero_fraction(x))


def cnn_forward_pass(x):
	"""
		Run through the feed forward
		3D Convolution Neural Network.

		Layers:
			3D ConvNet Layer 1 w/ Relu
			3D Maxpool Layer

			3D ConvNet Layer 2 w/ Relu
			3D Maxpool Layer

			Fully Connected Layer w/ Relu

			FC Output Layer that is returned and fed to Softmax
	"""
	with tf.variable_scope('conv1') as scope:
		conv1_filters = _variable_on_cpu('weights',
										 [3,3,3,1,32],
										 tf.contrib.layers.xavier_initializer())
		biases = _variable_on_cpu('biases',
								  [32],
								  tf.constant_initializer(0.0))
		conv = conv3d(samples,conv1_filters)
		logits = tf.nn.bias_add(conv,biases)
		conv1 = tf.nn.relu(logits, name=scope.name)
		#_activation_summary(conv1)
		conv1 = maxpool3d(conv1)
	# print(conv1.get_shape())
	with tf.variable_scope('conv2') as scope:
		conv2_filters = _variable_on_cpu('weights',
										 [3,3,3,32,64],
										 tf.contrib.layers.xavier_initializer())
		biases = _variable_on_cpu('biases',
								  [64],
								  tf.constant_initializer(0.0))
		conv = conv3d(conv1,conv2_filters)
		logits = tf.nn.bias_add(conv,biases)
		conv2 = tf.nn.relu(logits, name=scope.name)
		#_activation_summary(conv2)
		conv2 = maxpool3d(conv2)
	# print(conv2.get_shape())
	with tf.variable_scope('fc1') as scope:
		fc_reshape = tf.reshape(conv2,[-1, 54080])
		fc_weights = _variable_on_cpu('weights',
									  [54080,1024],
									  tf.contrib.layers.xavier_initializer())
		biases = _variable_on_cpu('biases',
								  [1024],
								  tf.constant_initializer(0.0))
		fc_logits = tf.nn.bias_add(tf.matmul(fc_reshape,fc_weights),biases)
		fc = tf.nn.relu(fc_logits)
		# Dropout could be applied here

	with tf.variable_scope('fc_out') as scope:
		fc_output_weights = _variable_on_cpu('weights',
									  		[1024,N_CLASSES],
									  		tf.contrib.layers.xavier_initializer())
		biases = _variable_on_cpu('biases',
								  [N_CLASSES],
								  tf.constant_initializer(0.0))
		output_logits = tf.nn.bias_add(tf.matmul(fc,fc_output_weights),biases)

	return output_logits



def main(*args):
	if FLAGS.samples:
		sample_data = get_sample_data(FLAGS.samples,FLAGS.labels)
	elif FLAGS.stage1:
		print("No function to handle stage1 data yet.")
	prediction = cnn_forward_pass(samples)
	# loss function
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
																  labels=labels))
	# backprop
	optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)
	print(optimizer)

if __name__ == "__main__":
    tf.app.run()
