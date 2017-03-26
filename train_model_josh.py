import os
import sys

import tensorflow as tf
import numpy as np

from utils import *
from ml_ops import *
from models import balanced as model

tf.app.flags.DEFINE_string("dataset", None, 'Path to dataset h5.')
tf.app.flags.DEFINE_string("sample_data", None, 'Path to sample data set directory.')
tf.app.flags.DEFINE_string("labels", None, 'Path to labels data set.')
tf.app.flags.DEFINE_string("model", None, 'Directory path to save out model and tensorboard files.')
# Globals
FLAGS = tf.app.flags.FLAGS
VALID_STEP = 700
VALID_CKPT_ONE = 10
VALID_CKPT_TWO = 200
# Using test sizes for now for faster debugging
IMAGE_HEIGHT = 140
IMAGE_WIDTH = 250
IMAGE_DEPTH = 325

# HP
BATCH_SIZE = 2
INITIAL_LEARNING_RATE = 5e-4
NUM_STEPS = 1000000
VALID_SPLIT = 0.2

# Tensorboard options
LAYERS_TO_GRAPH = ('conv1','conv2','conv3','conv4','conv5','conv6','fc1','fc2','fc3')

if FLAGS.dataset:
    dataset = Dataset(FLAGS.dataset,FLAGS.labels,valid_split=VALID_SPLIT)
else:
    # If no dataset provided, create random data (useful for testing)
    dataset = TestDataset(sample_path=FLAGS.sample_data,label_path=FLAGS.labels)

if not os.path.exists(FLAGS.model):
    os.makedirs(FLAGS.model)

if FLAGS.model[-1] == '/':
    model_name = FLAGS.model[:-1].split('/')[-1]
    model_path = os.path.join(FLAGS.model, model_name+'.ckpt')
else:
    model_name = FLAGS.model.split('/')[-1]
    model_path = os.path.join(FLAGS.model, model_name+'.ckpt')

# TF Placeholders
input_placeholder = tf.placeholder(tf.float32,[None,
                                               IMAGE_HEIGHT,
                                               IMAGE_WIDTH,
                                               IMAGE_DEPTH,
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
# Add loss to Tensorboard
tf.summary.scalar('loss', loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def main(*args):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(FLAGS.model,
                                       sess.graph)
        tf.global_variables_initializer().run()
        #graph_histograms(LAYERS_TO_GRAPH)
        print("Training...")
        training_losses = []
        best_valid = 99999999
        for step in range(NUM_STEPS):
            patient, label_batch, data_batch = dataset.get_batch(batch_size=BATCH_SIZE)
            #print(np.min(data_batch))
            #print(np.mean(data_batch))
            #print(np.max(data_batch))
            
            if step == 0:
                print("Data Shape:")
                print(data_batch.shape)
            
            data_batch = data_batch.reshape([-1,
                                             IMAGE_HEIGHT,
                                             IMAGE_WIDTH,
                                             IMAGE_DEPTH,
                                             1])
                                             
            feed_dict = {input_placeholder: data_batch,
                         labels_placeholder: label_batch,
                          }
            _, l, output, summary = sess.run([optimizer, loss, logits, merged],
                                             feed_dict=feed_dict)

            training_losses.append(l)
            #print(list(np.squeeze(output)))
            
            # Write summary object to Tensorboard
            # So far only writing training data
            writer.add_summary(summary, step)

            if step % VALID_STEP == 0 or step == 10 or step == 200:


                #print(list(np.squeeze(output)))

                print("Validating...")
                v_losses = []
                for vstep in range(100):
                    valid_patient, valid_label_batch, valid_data_batch = dataset.get_batch(train=False, batch_size=BATCH_SIZE)
                    valid_data_batch = valid_data_batch.reshape([-1,
                                                     IMAGE_HEIGHT,
                                                     IMAGE_WIDTH,
                                                     IMAGE_DEPTH,
                                                     1])
                    feed_dict = {input_placeholder: valid_data_batch,
                                labels_placeholder: valid_label_batch,
                                }
                    valid_l, output = sess.run([loss, logits],
                                                feed_dict=feed_dict)
                    v_losses.append(valid_l)
                    
                    if step == 0:
                        break
                
                v_term = ""
                if np.mean(v_losses) < best_valid:
                    best_valid = np.mean(v_losses)
                    saver.save(sess, model_path, step)
                    v_term = " I"
                        

                print("{} T: {} V: {}{}".format(step, np.mean(training_losses), np.mean(v_losses), v_term))
                training_losses = []

#            if step == VALID_CKPT_ONE:
#                valid_patient, valid_label_batch, valid_data_batch = dataset.get_batch(train=False)
#                feed_dict = {input_placeholder: valid_data_batch,
#                            labels_placeholder: valid_label_batch,
#                            }
#                valid_l, output = sess.run([loss, logits],
#                                            feed_dict=feed_dict)
#                print("Validation loss {}".format(l))
#                #print(list(np.squeeze(output)))
#
#            if step == VALID_CKPT_TWO:
#                valid_patient, valid_label_batch, valid_data_batch = dataset.get_batch(train=False)
#                feed_dict = {input_placeholder: valid_data_batch,
#                            labels_placeholder: valid_label_batch,
#                            }
#                valid_l, output = sess.run([loss, logits],
#                                            feed_dict=feed_dict)
#                print("Validation loss {}".format(l))
#                #print(list(np.squeeze(output)))

if __name__ == "__main__":
    tf.app.run()
