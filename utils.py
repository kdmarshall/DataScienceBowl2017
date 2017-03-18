import os
import numpy as np
import pandas as pd
import random

def get_array_from_dcm(filepath, dtype='uint8'):
    ds = dicom.read_file(filepath)
    np_array = ds.pixel_array.astype(dtype)
    return np_array

def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(a):
    return sum(a) / len(a)


class TestDataset(object):
    def __init__(self,sample_path=None,label_path=None):
        if sample_path:
            self.sample_path = sample_path
            self.label_path = label_path
            self.samples,self.labels = self._build_sample_set()
    
    def get_batch(self, batch_size):
        data = np.ones((batch_size, 16, 16, 16, 1)) * 0.001
        labels = np.ones((batch_size, 1))

        return data, labels

    def get_sample_batch(self):
        data_len = len(self.samples)
        batch_index = np.random.randint(0,high=data_len)
        return self.samples[batch_index], np.array([[self.labels[batch_index]]])

    def _build_sample_set(self):
        import dicom
        patients = [fname for fname in os.listdir(self.sample_path) if fname != '.DS_Store']
        labels_df = pd.read_csv(self.label_path, index_col='id')
        samples = []
        labels = []
        for patient in patients:
            try:
                label = labels_df.get_value(str(patient), 'cancer')
            except:
                print("Patient {} not found".format(patient))
                continue
            file_path = os.path.join(self.sample_path,patient)
            slices = [dicom.read_file(os.path.join(file_path,fname))
                        for fname in os.listdir(file_path)
                            if fname != '.DS_Store']
            slices.sort(key=lambda x: int(x.ImagePositionPatient[2]))
            slices = [np.array(each_slice.pixel_array).astype('uint8').reshape((512,512,1))
                        for each_slice in slices]
            samples.append(np.array(slices))
            labels.append(int(label))
        return samples, labels

class Dataset(object):
    def __init__(self, dir_path, labels_path):
    
        def match_labels():
            # take patient names and match them with the corresponding labels
            pass

        labels = []
        array_paths = []
        self.patient_nums = len(labels)

    def get_batch():
        patient_id = random.choice(patient_nums)
        label = self.labels[patient_id]
        array_path = self.array_paths[patient_id]

        gzipfile = gzip.GzipFile(os.path.join(root_p, path), 'r')
        arr = np.load(gzipfile)

        # Check shape




