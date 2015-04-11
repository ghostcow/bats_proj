"""
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
"""


## plan:
#   1. load flipping mat files n shit
#   2. do something with all the data??
#   3. ???
#   4. profit!

from Scientific.IO.NetCDF import *
import h5py
import os
import numpy as np
from nc_functions import *


  ## each column signifies a different classification task
                                                ## corresponding to columns [4 5 6 7 8 9 11] in the seqAnnotation matrix


def make_nc(mat_file, labels, nc_name, nc_path):

    ## bat .mat file for data
    mat = h5py.File(mat_file)
    concat_spectro = mat['concat_spectro'].value
    lengths = mat['lengths'].value
    sequence_count = int(mat['num_seq'].value[0][0])
    total_length = int(mat['total_length'].value[0][0])

    ## open new .nc file
    nc_path = os.path.join(nc_path, nc_name)
    nc_file = NetCDFFile(nc_path, 'w')

    ## create dimensions
    numSeqs_name = 'numSeqs'
    numSeqs_size = sequence_count
    nc_file.createDimension(numSeqs_name, numSeqs_size)

    numTimeSteps_name = 'numTimeSteps'
    numTimeSteps_size = total_length
    nc_file.createDimension(numTimeSteps_name, numTimeSteps_size)

    inputPattSize_name = 'inputPattSize'
    inputPattSize_size = 257
    nc_file.createDimension(inputPattSize_name, inputPattSize_size)

    maxSeqTagLength_name = 'maxSeqTagLength'
    maxSeqTagLength_size = int(np.ceil(np.log10(sequence_count))+1)
    nc_file.createDimension(maxSeqTagLength_name, maxSeqTagLength_size)

    numLabels_name = 'numLabels'
    numLabels_size = labels.shape[0]
    nc_file.createDimension(numLabels_name, numLabels_size)

    #######################################################
    ## create variables
    # varDims = (1, 2)
    # var1 = f.createVariable('varOne', 'i', varDims)
    seqTags_dims = (numSeqs_name, maxSeqTagLength_name)
    seqTags_var = nc_file.createVariable('seqTags', 'c', seqTags_dims)

    seqLengths_dims = (numSeqs_name,)
    seqLengths_var = nc_file.createVariable('seqLengths', 'i', seqLengths_dims)

    inputs_dims = (numTimeSteps_name, inputPattSize_name)
    inputs_var = nc_file.createVariable('inputs', 'f', inputs_dims)

    targetClasses_dims = (numTimeSteps_name,)
    targetClasses_vars = nc_file.createVariable('targetClasses', 'i', targetClasses_dims)

    #######################################################
    ## fill variables
    for i in xrange(total_length):
        num = str(i)
        num_len = len(num)
        seqTags_var[i, 0:num_len] = num

    lengths = lengths.astype(np.int32)
    seqLengths_var[:] = lengths

    inputs_var[:] = concat_spectro

    labels = stretch_labels(labels, lengths)
    targetClasses_vars[:] = labels

    #######################################################
    ## end nc creation, clean up
    nc_file.flush(); nc_file.close()
    mat.close()
    #######################################################