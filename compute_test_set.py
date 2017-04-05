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
dir_path = os.path.dirname(os.path.realpath(__file__))

model = None
models_func_list = dir(models)
for model_func in models_func_list:
  if model_func == FLAGS.model:
    model = getattr(models, model_func)
    print("Using model {}".format(model_func))

if not model:
  sys.exit("Cannot determine model name {}".format(FLAGS.model))

submission_file_path = os.path.join(dir_path,'submissions',FLAGS.model+'.csv')
submission_file = open(submission_file_path,'w')
# Using test sizes for now for faster debugging
IMAGE_HEIGHT = 140
IMAGE_WIDTH = 250
IMAGE_DEPTH = 325

dataset = Dataset(FLAGS.dataset, FLAGS.labels, valid_split=0.5)

# TF Placeholders
input_placeholder = tf.placeholder(tf.float32,[None,
                                               IMAGE_HEIGHT,
                                               IMAGE_WIDTH,
                                               IMAGE_DEPTH,
                                               1])

_logits = model(input_placeholder)
logits = tf.nn.sigmoid(_logits)

def main(*args):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, FLAGS.ckpt)

        submission_file.write('id,cancer\n')
        for patiend_id, data_batch in dataset.inference_iteritems():
            
            data_batch = data_batch.reshape([-1,
                                             IMAGE_HEIGHT,
                                             IMAGE_WIDTH,
                                             IMAGE_DEPTH,
                                             1])
                                             
            feed_dict = {input_placeholder: data_batch}
            logits = sess.run([logits], feed_dict=feed_dict)[0]
            sq_logits = np.squeeze(logits)
            submission_file.write(patiend_id+","+str(sq_logits[0])+"\n")
        submission_file.close()


if __name__ == "__main__":
    tf.app.run()
