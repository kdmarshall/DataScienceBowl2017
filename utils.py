import os
import numpy as np
import pandas as pd
import random
import gzip
import math

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
    def __init__(self, dir_path, labels_path, valid_split=0.2):
        self.valid_split = valid_split
        self.sample_dir_path = dir_path
        paths = os.listdir(dir_path)
        patient_paths = [os.path.join(dir_path, x) for x in paths]
        labels_df = pd.read_csv(labels_path, index_col='id')
        # We need to match patients with labels
        self.patients = {}
        patients_not_found = []
        for patient in patient_paths:
            patient_id = patient.split('/')[-1].split('.')[0]
            try:
                label = labels_df.get_value(str(patient_id), 'cancer')
            except:
                # print("Patient {} not found".format(patient_id))
                patients_not_found.append(patient_id)
                continue
            self.patients[patient_id] = int(label)
        # print("Did not find {} out of {} total patients".format(len(patients_not_found),len(patient_paths)))
        self.patient_nums = len(self.patients)
        patient_keys = list(self.patients.keys())
        np.random.shuffle(patient_keys)
        num_valid = math.ceil(self.patient_nums * self.valid_split)
        num_train = self.patient_nums - num_valid
        self.train_set, self.valid_set = patient_keys[:num_train], patient_keys[num_train:]
        
    def transform(self, arr, get_stats=False):
        #print(arr.shape)
        # Check shape. TODO: These should be abstracted into a single function (DRY)
        if arr.shape[0] < 140:
            am = 140 - arr.shape[0]
            pad = np.random.randint(am)
            arr = np.lib.pad(arr, ((pad, am-pad),(0, 0),(0, 0)), 'constant', constant_values=-2000)
        
        if arr.shape[1] < 250:
            am = 250 - arr.shape[1]
            pad = np.random.randint(am)
            arr = np.lib.pad(arr, ((0, 0),(pad, am-pad),(0, 0)), 'constant', constant_values=-2000)
        
        if arr.shape[2] < 325:
            am = 325 - arr.shape[2]
            pad = np.random.randint(am)
            arr = np.lib.pad(arr, ((0, 0),(0, 0),(pad, am-pad)), 'constant', constant_values=-2000)
        
        
        if arr.shape[0] > 140:
            sl = np.random.randint(arr.shape[0] - 140)
            arr = arr[sl:arr.shape[0] - ((arr.shape[0] - 140) - sl), :, :]
            
        if arr.shape[1] > 250:
            sl = np.random.randint(arr.shape[1] - 250)
            arr = arr[:, sl:arr.shape[1] - ((arr.shape[1] - 250) - sl), :]
        
        if arr.shape[2] > 325:
            sl = np.random.randint(arr.shape[2] - 325)
            arr = arr[:, :, sl:arr.shape[2] - ((arr.shape[2] - 325) - sl)]

        # Normalize values
        mask = arr<-800
        mask += arr>400
        
        # TODO: Print max/min to see that ranges are as expected
        #print(np.max(arr))
        #print(np.min(arr))
        if not get_stats:
            mean = -521.740256986
            std = 289.3945501514394
            arr = (arr.astype('float32') - mean)/std
            arr[mask] = 0
            return arr
        else:
            values = np.extract(np.invert(mask), arr)
            return values
            
    
    def get_batch(self, train=True):
        data_set = self.train_set if train else self.valid_set
        patient = random.choice(data_set)
        label = self.patients[patient]
        gz_file_path = os.path.join(self.sample_dir_path,patient + '.npy.gz')
        gzipfile = gzip.GzipFile(gz_file_path, 'r')
        sample_arr = np.load(gzipfile)
        label_arr = np.array(label).reshape([-1,1])
        sample_arr = self.transform(sample_arr)
        return patient, label_arr, sample_arr

    def compute_statistics(self):
        
        data_set = sorted(self.train_set)
        values = []
        print("Computing stats for {} patients...".format(len(data_set)))
        n = 0
        mean = 0
        M2 = 0
        for patient in data_set:
        
            gz_file_path = os.path.join(self.sample_dir_path,patient + '.npy.gz')
            gzipfile = gzip.GzipFile(gz_file_path, 'r')
            sample_arr = np.load(gzipfile)

            values += list(self.transform(sample_arr, True))
        
            for x in self.transform(sample_arr, True):
                n = n + 1
                delta = x - mean
                mean = mean + delta/n
                M2 = M2 + delta*(x - mean)
            print(mean)
            print(math.sqrt(M2/(n - 1)))
    
        #m = np.mean(values)
        #s = np.std(values)
        
        print(mean)
        print(math.sqrt(M2/(n - 1)))

        #print(m)
        #print(s)
        print("Total voxels included: {}".format(len(values)))

#    def get_batch():
#        patient_id = random.choice(self.patients.keys())
#        label = self.patients[patient_id]['label']
#        _path = self.patients[patient_id]['path']
#
#        gzipfile = gzip.GzipFile(_path, 'r')
#        arr = np.load(gzipfile)




if __name__ == '__main__':
    # Perform tests on the dataset
    dataset = Dataset('/Users/Peace/Desktop/from_gcloud/processed', 'test')




