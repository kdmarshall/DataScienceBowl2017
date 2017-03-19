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
tf.app.flags.DEFINE_string("model", None, 'Directory path to save out model and tensorboard files.')
# Globals
FLAGS = tf.app.flags.FLAGS
# Using test sizes for now for faster debugging
IMAGE_WIDTH = 140
IMAGE_HEIGHT = 250
IMAGE_DEPTH = 325

# HP
BATCH_SIZE = 2
INITIAL_LEARNING_RATE = 1e-3
NUM_STEPS = 100

if FLAGS.dataset:
    dataset = Dataset(FLAGS.dataset,FLAGS.labels)
else:
    # If no dataset provided, create random data (useful for testing)
    dataset = TestDataset(sample_path=FLAGS.sample_data,label_path=FLAGS.labels)

if FLAGS.model:
    if not os.path.exists(FLAGS.model):
        os.makedirs(FLAGS.model)

# TF Placeholders
input_placeholder = tf.placeholder(tf.float32,[None, 
                                               IMAGE_DEPTH,
                                               IMAGE_HEIGHT,
                                               IMAGE_WIDTH,
                                               1])

# Sample data TF Placeholders
# sample_input_placeholder = tf.placeholder(tf.float32,[None, 
#                                                        None,
#                                                        512,
#                                                        512,
#                                                        1])
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
            patient, label_batch, data_batch = dataset.get_batch()
            # data_batch, labels_batch = dataset.get_sample_batch()
            print("Data Shape:")
            print(data_batch.shape)
            print("Label Shape:")
            print(label_batch.shape)
            sys.exit(0)
            feed_dict = {input_placeholder: data_batch,
                         labels_placeholder: label_batch,
                          }
            _, l, output = sess.run([optimizer, loss, logits],
                                    feed_dict=feed_dict)
        
            if step % 10 == 0:
                print(l)
                
                print(list(np.squeeze(output)))

if __name__ == "__main__":
    tf.app.run()
