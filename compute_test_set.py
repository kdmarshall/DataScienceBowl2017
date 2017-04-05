import os
import sys

import tensorflow as tf
import numpy as np

from utils import *
from ml_ops import *
from models import recycled as model

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset h5.')
tf.app.flags.DEFINE_string("labels", None, 'Path to labels data set.')
tf.app.flags.DEFINE_string("model", None, 'Directory path to save out model and tensorboard files.')

# Globals
FLAGS = tf.app.flags.FLAGS

# Using test sizes for now for faster debugging
IMAGE_HEIGHT = 140
IMAGE_WIDTH = 250
IMAGE_DEPTH = 325

dataset = Dataset(FLAGS.dataset, FLAGS.labels, valid_split=VALID_SPLIT)


#if FLAGS.model[-1] == '/':
#    model_name = FLAGS.model[:-1].split('/')[-1]
#    model_path = os.path.join(FLAGS.model, model_name+'.ckpt')
#else:
#    model_name = FLAGS.model.split('/')[-1]
#    model_path = os.path.join(FLAGS.model, model_name+'.ckpt')

# TF Placeholders
input_placeholder = tf.placeholder(tf.float32,[None,
                                               IMAGE_HEIGHT,
                                               IMAGE_WIDTH,
                                               IMAGE_DEPTH,
                                               1])

learning_rate = tf.Variable(INITIAL_LEARNING_RATE, trainable=False)

_logits = model(input_placeholder)
logits = tf.nn.sigmoid(_logits)

def main(*args):
    with tf.Session() as sess:
        saver = tf.train.Saver()

        for step in range(NUM_STEPS):
            ###patient, label_batch, data_batch = dataset.get_batch(batch_size=BATCH_SIZE)
            
            data_batch = data_batch.reshape([-1,
                                             IMAGE_HEIGHT,
                                             IMAGE_WIDTH,
                                             IMAGE_DEPTH,
                                             1])
                                             
            feed_dict = {input_placeholder: data_batch}
            
            logits = sess.run([logits], feed_dict=feed_dict)[0]


if __name__ == "__main__":
    tf.app.run()
