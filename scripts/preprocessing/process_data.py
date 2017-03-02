import dicom
import numpy as np
import pandas as pd
from PIL import Image
import cv2

HM_SLICES = 20

# TODO: TF flags should be changed to args
tf.app.flags.DEFINE_string("stage1",'','Full training examples set.')
tf.app.flags.DEFINE_string("samples",'','Sample training examples set.')
tf.app.flags.DEFINE_string("labels",'','Full training labels set.')

if FLAGS.samples:
    sample_data = get_sample_data(FLAGS.samples,FLAGS.labels)
elif FLAGS.stage1:
    print("No function to handle stage1 data yet.")

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

# TODO: Save out dataset (h5)
