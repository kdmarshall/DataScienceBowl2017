"""
Example: python /Users/Peace/Desktop/data_processor.py /Users/Peace/Desktop/Joshs/stage1 /Users/Peace/Desktop/output_phase1
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage as ndimage
import matplotlib
matplotlib.use('TkAgg') # For OSX
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

from skimage import measure, morphology, segmentation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from common import *

parser = argparse.ArgumentParser()
parser.add_argument("input_dir")
parser.add_argument("output_dir")
args = parser.parse_args()

# Some constants 
INPUT_FOLDER = args.input_dir # stage1 dir
OUTPUT_DIR = args.output_dir # phase1 dir
patients = os.listdir(INPUT_FOLDER)
patients.sort()

patients = [x for x in patients if x[0] != '.'] # DS_STORE

# Create output dirs
for patient in patients:
    outdir = os.path.join(OUTPUT_DIR, patient)
    if not os.path.exists(outdir):
        os.mkdir(outdir)



test_patient_scans = load_scan(os.path.join(INPUT_FOLDER, patients[8]))
test_patient_images = get_pixels_hu(test_patient_scans)

print(test_patient_scans[20].SliceThickness)
print(test_patient_scans[20].PixelSpacing)


test_segmented, _, _, _, _, _, _, _ = seperate_lungs(test_patient_images[65])
test_segmented[test_segmented == -2000] = 0
print(np.max(test_segmented))
print(np.min(test_segmented))

resample(image, scan, new_spacing=[1,1,1])

sdfsdf




def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

first_patient = load_scan(INPUT_FOLDER + patients[0])
first_patient_pixels = get_pixels_hu(first_patient)

print(first_patient_pixels.shape)
Image.fromarray(first_patient_pixels[64
]).save('/Users/Peace/Desktop/example0.png')
sdfsf

pix_resampled, spacing = resample(first_patient_pixels, first_patient, [.1,.1,.1])
print("Shape before resampling\t", first_patient_pixels.shape)
print("Shape after resampling\t", pix_resampled.shape)

segmented_lungs = segment_lung_mask(pix_resampled, False)
segmented_lungs_fill = segment_lung_mask(pix_resampled, True)



MIN_BOUND = -1000.0
MAX_BOUND = 400.0
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image

PIXEL_MEAN = 0.25

def zero_center(image):
    image = image - PIXEL_MEAN
    return image





