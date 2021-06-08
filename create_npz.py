#!/usr/bin/python

import sys
import numpy as np
import nibabel as nib
import os

#[1] file path, [2] file name, [3] file destination
filename = os.path.join(sys.argv[1], sys.argv[2]+".nii.gz")
img = nib.load(filename)
data = np.array(img.dataobj)
print(data.shape)
#f=data.files
s = data.shape[0]
#a = data[48:(s-48), 31:(s-33), 3:(s-29)] 
#whole volume
a = data[55:199, 95:207, 66:162]
#a part of volume due to gb constraints
print(a.shape)
a = (a/255).astype('float32')
print(a.dtype)
str = sys.argv[3]+"/"+sys.argv[2]+'.npz'
np.savez_compressed(str, vol=a)
