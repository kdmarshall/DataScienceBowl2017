import numpy as np
import dicom


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