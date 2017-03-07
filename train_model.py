import os
import sys

import tensorflow as tf
import numpy as np

from utils import *
from ml_ops import *
from models import baseline_model as model

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset h5.')
tf.app.flags.DEFINE_string("sample_data", None, 'Path to sample data set directory.')
tf.app.flags.DEFINE_string("labels", None, 'Path to labels data set.')

# Globals
FLAGS = tf.app.flags.FLAGS
# Using test sizes for now for faster debugging
IMAGE_WIDTH = 16#256
IMAGE_HEIGHT = 16#256
IMAGE_DEPTH = 16#120

# HP
BATCH_SIZE = 2
INITIAL_LEARNING_RATE = 1e-3
NUM_STEPS = 100

if FLAGS.dataset:
    # TODO
    pass
else:
    # If no dataset provided, create random data (useful for testing)
    dataset = TestDataset(sample_path=FLAGS.sample_data)

# TF Placeholders
# input_placeholder = tf.placeholder(tf.float32,[None, 
#                                                IMAGE_DEPTH,
#                                                IMAGE_HEIGHT,
#                                                IMAGE_WIDTH,
#                                                1])

# Sample data TF Placeholders
input_placeholder = tf.placeholder(tf.float32,[None, 
                                               None,
                                               512,
                                               512,
                                               1])
labels_placeholder = tf.placeholder(tf.float32, [None, 1]) # 1 class: 0 or 1
#training_placeholder = tf.placeholder(tf.bool)
learning_rate = tf.Variable(INITIAL_LEARNING_RATE, trainable=False)

_logits = model(input_placeholder)
logits = tf.nn.sigmoid(_logits)

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=_logits,
                                                    labels=labels_placeholder))

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)


def main(*args):
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print("Training...")
        for step in range(NUM_STEPS):
            # data_batch, labels_batch = dataset.get_batch(BATCH_SIZE)
            data_batch, labels_batch = dataset.get_sample_batch()
            print(data_batch.shape)
            sys.exit(0)
            feed_dict = {input_placeholder: data_batch,
                         labels_placeholder: labels_batch,
                          }
            _, l, output = sess.run([optimizer, loss, logits],
                                    feed_dict=feed_dict)
        
            if step % 10 == 0:
                print(l)
                
                print(list(np.squeeze(output)))

if __name__ == "__main__":
    tf.app.run()
