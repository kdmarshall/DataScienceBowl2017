import dicom
import numpy as np
import pandas as pd
import tensorflow as tf
# from PIL import Image
# import cv2
import os
import math

from utils import *

HM_SLICES = 20
# IMAGE_WIDTH = 256
# IMAGE_HEIGHT = 256
IMAGE_DEPTH = 120

# TODO: TF flags should be changed to args
tf.app.flags.DEFINE_string("stage1",'','Full training examples set.')
tf.app.flags.DEFINE_string("samples",'','Sample training examples set.')
tf.app.flags.DEFINE_string("labels",'','Full training labels set.')

FLAGS = tf.app.flags.FLAGS

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
		# slices = [np.array(each_slice.pixel_array).astype('uint8')
		# 			for each_slice in slices]
		# print(len(slices))
		start = float(slices[0].SliceLocation)
		# print("Start {}".format(start))
		finish = float(slices[-1].SliceLocation)
		# print("Finish {}".format(finish))
		depth =  finish - start
		print("Depth {}".format(abs(depth)))
		# return
		# print("**********")

def main(*args):
	if FLAGS.samples:
		sample_data = get_sample_data(FLAGS.samples,FLAGS.labels)
	elif FLAGS.stage1:
		print("No function to handle stage1 data yet.")
	# TODO: Save out dataset (h5)

if __name__ == "__main__":
    tf.app.run()
