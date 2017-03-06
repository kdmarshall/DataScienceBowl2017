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
args = parser.parse_args()

INPUT_FOLDER = args.input_dir # stage1 dir
patients = os.listdir(INPUT_FOLDER)
patients.sort()

patients = [x for x in patients if x[0] != '.'] # DS_STORE

thickness_vals = []
spacing_vals = []
spacing_diffs = []

for patient in patients:
    try:
        patient_scan = load_scan(os.path.join(INPUT_FOLDER, patient))
    except:
        print("Patient skipped: {}".format(patient))
        continue
    
    for frame in patient_scan:
        thickness_vals.append(frame.SliceThickness)
        spacing_vals += list(frame.PixelSpacing)
        spacing_diffs.append(frame.PixelSpacing[0] - frame.PixelSpacing[1])
        break # Only looking at one frame per scan; assuming consistency within patient


plt.hist(thickness_vals, bins=60)
plt.ylabel('Thicknesses')
plt.savefig(os.path.expanduser('~/Desktop/thick_dist.png'))
plt.clf()

plt.hist(spacing_vals, bins=60)
plt.ylabel('Spacing')
plt.savefig(os.path.expanduser('~/Desktop/space_dist.png'))
plt.clf()

plt.hist(spacing_diffs, bins=60)
plt.ylabel('Spacing Diffs')
plt.savefig(os.path.expanduser('~/Desktop/diffs_dist.png'))
plt.clf()


