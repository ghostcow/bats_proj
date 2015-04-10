'''
+-----------------------------------------------------------------------------+
| 5.1 General structure                                                       |
+-----------------------------------------------------------------------------+
The data files need to contain the following dimensions:
* numSeqs         = Number of sequences
* numTimesteps    = Total number of timesteps
* inputPattSize   = Size of each input pattern (= number of input neurons)
* maxSeqTagLength = Maximum length of a sequence tag

The required variables are:
* char seqTags(numSeqs, maxSeqTagLength)    = Tag (name) for each sequence
* int seqLengths(numSeqs)                   = Length of each sequence
* float inputs(numTimesteps, inputPattSize) = Input Patterns

+-----------------------------------------------------------------------------+
| 5.2 Regression tasks                                                        |
+-----------------------------------------------------------------------------+
Additional dimensions:
* targetPattSize = Size of each output pattern (= number of output neurons)

Additional variables:
* float targetPatterns(numTimesteps, targetPattSize) = Target patterns

Optional variables:
* float outputMeans(targetPattSize) = estimated means of outputs
* float outputStdevs(targetPattSize) = estimated standard deviations of outputs
  (used to revert standardization of outputs in forward pass mode)

+-----------------------------------------------------------------------------+
| 5.3 Classification tasks                                                    |
+-----------------------------------------------------------------------------+
Additional dimensions:
* numLabels = Number of target classes

Additional variables:
* int targetClasses(numTimesteps) = Target classes (one for each timestep)
'''


## plan:
#   1. load flipping mat files n shit
#   2. do something with all the data??
#   3. ???
#   4. profit!

from Scientific.IO.NetCDF import *
import h5py
import os
import numpy as np

mat = h5py.File('/home/lioruzan/bats_proj/data/spectrograms/500_250_concat.mat')

concat_spectro = mat['concat_spectro'].value
lengths = mat['lengths'].value.transpose()
sequence_count = int(mat['num_seq'].value[0][0])
total_length = int(mat['total_length'].value[0][0])
sequence_labels = mat['sequence_labels'].value  ## each column signifies a different classification task
                                                ## corresponding to columns [4 5 6 7 8 9 11] in the seqAnnotation matrix


def process_labels(matrix, column):
    map = {}
    vec = matrix[:, column]

    dtype = matrix.dtype
    vec_shape = vec.shape

    new_vec = np.zeros(vec_shape, dtype)
    c = 0
    for i in xrange(len(vec)):

        old_num = vec[i]
        if old_num not in map.keys():

            map[old_num] = c
            c += 1
        new_vec[i] = map[old_num]
    matrix[:, column] = new_vec
    return map


label_maps = [process_labels(sequence_labels, i) for i in xrange(sequence_labels.shape[0])]

# open new .nc file
f = NetCDFFile('/home/lioruzan/bats_proj/data/nc_files/test.nc', 'w')


# create dimensions
numSeqs_name = 'numSeqs'
numSeqs_size = sequence_count
f.createDimension(numSeqs_name, numSeqs_size)

numTimeSteps_name = 'numTimeSteps'
numTimeSteps_size = total_length
f.createDimension(numTimeSteps_name, numTimeSteps_size)

inputPattSize_name = 'inputPattSize'
inputPattSize_size = 257
f.createDimension(inputPattSize_name, inputPattSize_size)

maxSeqTagLength_name = 'maxSeqTagLength'
maxSeqTagLength_size = int(np.ceil(np.log10(sequence_count))+1)
f.createDimension(maxSeqTagLength_name, maxSeqTagLength_size)

numLabels_name = 'numLabels'
numLabels_size = ? # update, depends on classification task
f.createDimension(numLabels_name, numLabels_size)

# create variables
# varDims = (1, 2)
# var1 = f.createVariable('varOne', 'i', varDims)
seqTags_dims = (numSeqs_name, maxSeqTagLength_name)
seqTags_var = f.createVariable('seqTags', 'c', seqTags_dims)

seqLengths_dims = (numSeqs_name,)
seqLengths_var = f.createVariable('seqLengths', 'i', seqTags_dims)

inputs_dims = (numTimeSteps_name, inputPattSize_name)
inputs_var = f.createVariable('inputs', 'f', inputs_dims)

targetClasses_dims = (numTimeSteps_name,)
targetClasses_vars = f.createVariable('targetClasses', 'i', targetClasses_dims)