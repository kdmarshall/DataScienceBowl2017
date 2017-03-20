import numpy as np
import tensorflow as tf
import dicom


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

def prelu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg

def convolve(input, i_units, o_units, name, kernel=3, activate=True, pool=True):
    with tf.variable_scope(name):
        conv_weights = tf.get_variable("w", [kernel, kernel, kernel, i_units,
                                               o_units],
                        initializer=tf.contrib.layers.xavier_initializer_conv2d()) # Note: 3d init not available. Think about this more.
        conv_bias = tf.get_variable("b",
                                    initializer=tf.constant(0., shape=[o_units]))
        pre = tf.nn.conv3d(input, conv_weights,
                                strides=[1, 1, 1, 1, 1], padding='SAME') + conv_bias
                                
        if activate:
            h_conv1 = prelu(pre)
        else:
            h_conv1 = pre

    if pool:
        pooled = tf.nn.max_pool3d(h_conv1,
							ksize=[1, 2, 2, 2, 1],
							strides=[1, 2, 2, 2, 1],
							padding='SAME')
    else:
        pooled = h_conv1

    return pooled

def fc_layer(input, o_units, name, activate=True):
    batch_size = tf.shape(input)[0]

    input_reshaped = tf.reshape(input, [batch_size, -1])

    in_size = np.prod(input.get_shape().as_list()[1:])
    fc_weights = tf.get_variable(name, [in_size, o_units],
                    initializer=tf.contrib.layers.xavier_initializer())
    fc_bias = tf.get_variable("{0}_b".format(name),
                    initializer=tf.constant(0., shape=[o_units]))

    pre = tf.nn.bias_add(tf.matmul(input_reshaped, fc_weights), fc_bias)

    if activate:
        h_fc = prelu(pre)
    else:
        h_fc = pre

    return h_fc

def graph_histograms(layer_names):
	for name in layer_names:
		if 'conv' in name:
			with tf.variable_scope(name) as scope_conv: 
				tf.get_variable_scope().reuse_variables()
				bias = tf.get_variable('b')
				tf.summary.histogram(bias.name, bias)
				weights = tf.get_variable('w')
				tf.summary.histogram(weights.name, weights)
		elif 'fc' in name:
			tf.get_variable_scope().reuse_variables()
			fc = tf.get_variable(name)
			tf.summary.histogram(fc.name, fc)
			fc_b = tf.get_variable(name+"_b")
			tf.summary.histogram(fc_b.name, fc_b)




