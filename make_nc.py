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


from Scientific.IO.NetCDF import *
import os
import numpy as np
from nc_functions import *


# don't forget to reshape 'lengths' to vector of dimension (dim,)
# def make_nc(concat_spectro, labels, num_labels, lengths, nc_name, nc_path):
#
#     ## open new .nc file
#     nc_path = os.path.join(nc_path, nc_name)
#     nc_file = NetCDFFile(nc_path, 'w')
#
#     ## create dimensions
#     sequence_count = len(lengths)
#     numSeqs_name = 'numSeqs'
#     numSeqs_size = sequence_count
#     nc_file.createDimension(numSeqs_name, numSeqs_size)
#
#     total_length = int(sum(lengths))
#     numTimeSteps_name = 'numTimeSteps'
#     numTimeSteps_size = total_length
#     nc_file.createDimension(numTimeSteps_name, numTimeSteps_size)
#
#     inputPattSize_name = 'inputPattSize'
#     inputPattSize_size = concat_spectro.shape[1]
#     nc_file.createDimension(inputPattSize_name, inputPattSize_size)
#
#     maxSeqTagLength_name = 'maxSeqTagLength'
#     maxSeqTagLength_size = int(np.ceil(np.log10(sequence_count)) + 1)
#     nc_file.createDimension(maxSeqTagLength_name, maxSeqTagLength_size)
#
#     numLabels_name = 'numLabels'
#     numLabels_size = num_labels
#     nc_file.createDimension(numLabels_name, numLabels_size)
#
#     #######################################################
#     ## create variables
#     # varDims = (1, 2)
#     # var1 = f.createVariable('varOne', 'i', varDims)
#     seqTags_dims = (numSeqs_name, maxSeqTagLength_name)
#     seqTags_var = nc_file.createVariable('seqTags', 'c', seqTags_dims)
#
#     seqLengths_dims = (numSeqs_name,)
#     seqLengths_var = nc_file.createVariable('seqLengths', 'i', seqLengths_dims)
#
#     inputs_dims = (numTimeSteps_name, inputPattSize_name)
#     inputs_var = nc_file.createVariable('inputs', 'f', inputs_dims)
#
#     targetClasses_dims = (numTimeSteps_name,)
#     targetClasses_vars = nc_file.createVariable('targetClasses', 'i', targetClasses_dims)
#
#     #######################################################
#     ## fill variables
#     for i in xrange(numSeqs_size):
#         num = str(i)
#         num_len = len(num)
#         seqTags_var[i, 0:num_len] = num
#
#     seqLengths_var[:] = lengths
#
#     inputs_var[:] = concat_spectro
#
#     labels = stretch_labels(labels, lengths)
#     targetClasses_vars[:] = labels
#
#     #######################################################
#     ## end nc creation, clean up
#     nc_file.close()


def make_nc(concat_spectro, labels, num_labels, lengths, nc_name, nc_path):

    ######################################################
    # calculate dimensions for nc file

    numSeqs = len(lengths)

    numTimeSteps = int(sum(lengths))

    inputPattSize = concat_spectro.shape[1]

    maxSeqTagLength = int(np.ceil(np.log10(numSeqs)) + 1)

    numLabels = num_labels

    #######################################################
    # create data for nc file variables

    seqTags_dims = (numSeqs, maxSeqTagLength)
    seqTags = np.zeros(seqTags_dims, dtype='S1')
    for i in xrange(numSeqs):
        num = str(i)
        num_len = len(num)
        seqTags[i, 0:num_len] = num

    seqLengths_dims = numSeqs
    seqLengths = np.zeros(seqLengths_dims, dtype='int32')
    seqLengths[:] = lengths

    inputs_dims = (numTimeSteps, inputPattSize)
    inputs = np.zeros(inputs_dims, dtype='float32')
    inputs[:] = concat_spectro

    targetClasses_dims = numTimeSteps
    targetClasses = np.zeros(targetClasses_dims, dtype='int32')
    labels = stretch_labels(labels, lengths)
    targetClasses[:] = labels

    #######################################################
    # create nc file
    _make_nc(
        nc_path, nc_name, numSeqs, numTimeSteps,
        inputPattSize, maxSeqTagLength, numLabels,
        seqTags, seqLengths, inputs, targetClasses)


def _make_nc(
        nc_path, nc_name, numSeqs, numTimeSteps,
        inputPattSize, maxSeqTagLength, numLabels,
        seqTags, seqLengths, inputs, targetClasses):

    #######################################################
    # open new .nc file

    nc_path = os.path.join(nc_path, nc_name)
    nc_file = NetCDFFile(nc_path, 'w')

    #######################################################
    # create dimensions

    numSeqs_name = 'numSeqs'
    numSeqs_size = numSeqs
    nc_file.createDimension(numSeqs_name, numSeqs_size)

    numTimeSteps_name = 'numTimeSteps'
    numTimeSteps_size = numTimeSteps
    nc_file.createDimension(numTimeSteps_name, numTimeSteps_size)

    inputPattSize_name = 'inputPattSize'
    inputPattSize_size = inputPattSize
    nc_file.createDimension(inputPattSize_name, inputPattSize_size)

    maxSeqTagLength_name = 'maxSeqTagLength'
    maxSeqTagLength_size = maxSeqTagLength
    nc_file.createDimension(maxSeqTagLength_name, maxSeqTagLength_size)

    numLabels_name = 'numLabels'
    numLabels_size = numLabels
    nc_file.createDimension(numLabels_name, numLabels_size)

    #######################################################
    # create variables in nc file
    #
    # for example:
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
    # fill variables

    seqTags_var[:] = seqTags
    seqLengths_var[:] = seqLengths
    inputs_var[:] = inputs
    targetClasses_vars[:] = targetClasses

    #######################################################
    # end nc creation, clean up

    nc_file.close()


# create dummy nc file for forward feeding a network with currennt.
# dummy in the sense that it replaces the numLabels dimension with the number of output units
def create_feedforward_dummy_nc(nc_path, old_nc_name, new_nc_name, num_output_units):
    print os.path.join(nc_path, old_nc_name)
    old = NetCDFFile(os.path.join(nc_path, old_nc_name))

    numSeqs = old.dimensions['numSeqs']
    numTimeSteps = old.dimensions['numTimeSteps']
    inputPattSize = old.dimensions['inputPattSize']
    maxSeqTagLength = old.dimensions['maxSeqTagLength']

    seqTags = old.variables['seqTags'].getValue()
    seqLengths = old.variables['seqLengths'].getValue()
    inputs = old.variables['inputs'].getValue()
    targetClasses = old.variables['targetClasses'].getValue()
    old.close()

    _make_nc(
        nc_path, new_nc_name, numSeqs, numTimeSteps,
        inputPattSize, maxSeqTagLength, num_output_units,
        seqTags, seqLengths, inputs, targetClasses)