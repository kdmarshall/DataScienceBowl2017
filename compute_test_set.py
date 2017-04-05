import os
import sys

import tensorflow as tf
import numpy as np

from utils import *
from ml_ops import *
import models

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset h5.')
tf.app.flags.DEFINE_string("labels", None, 'Path to labels data set.')
tf.app.flags.DEFINE_string("ckpt", None, 'Directory path to saved checkpoint file.')
tf.app.flags.DEFINE_string("model", None, 'Name of model to use.')

# Globals
FLAGS = tf.app.flags.FLAGS

# if FLAGS.model:
#     from models import recycled as model

model = None
models_func_list = dir(models)
for model_func in models_func_list:
  if model_func == FLAGS.model:
    model = getattr(models, model_func)
    print("Using model {}".format(model_func))

if not model:
  sys.exit("Cannot determine model name {}".format(FLAGS.model))

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
        
        for patiend_id, data_batch in dataset.inference_iteritems():
            
            data_batch = data_batch.reshape([-1,
                                             IMAGE_HEIGHT,
                                             IMAGE_WIDTH,
                                             IMAGE_DEPTH,
                                             1])
                                             
            feed_dict = {input_placeholder: data_batch}
            
            logits = sess.run([logits], feed_dict=feed_dict)[0]


if __name__ == "__main__":
    tf.app.run()
