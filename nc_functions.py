__author__ = 'lioruzan'

import numpy as np
from Scientific.IO.NetCDF import *


"""
bats .mat file format:
concat_spectro - concatenated inputs, dimensions are (numTimesteps, inputPattSize)
lengths - vector of sequence lengths, dimensions are (numSeqs, 1)
sequence_count - number of sequences
total_length - number of time steps
sequence_labels - class labels, dimensions are (numSeqs, numClassificationTasks),
                    where numClassificationTasks is the different options to label the sequences.
                    each column signifies a different classification task, corresponding to
                    columns [4 5 6 7 8 9 11] in the seqAnnotation matrix.
                    the matrix columns are not normalized.
"""


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


def process_label_matrix(label_matrix):
    label_maps = [process_labels(label_matrix, i) for i in xrange(label_matrix.shape[1])]
    return label_maps


# turn sequence labels to timestep labels
def stretch_labels(labels, lengths):
    length = int(sum(lengths))
    out_labels = np.zeros((length,), dtype='int32')
    s = 0
    for i in xrange(len(labels)):
        e = lengths[i]
        out_labels[s:s+e] = np.zeros((e,), 'int32') + labels[i]
        s += e
    return out_labels


# don't forget to reshape sequence_lengths to be one-dimensional
# 0 <= n < matrix.shape[0]
def get_sequence_from_matrix(n, matrix, sequence_lengths):
    if n >= len(sequence_lengths):
        return None
    s = sum(sequence_lengths[0:n])
    e = sequence_lengths[n]
    return matrix[s:s+e]


def subsample_sequence_matrix(sample_indices, sequence_matrix, sequence_lengths):
    sequences = [get_sequence_from_matrix(i, sequence_matrix, sequence_lengths) for i in sample_indices]
    new_seq_mat = np.vstack(sequences)
    return new_seq_mat


# also takes only vector one-dimensional sequence_lengths, labels.
def subsample_sequences(sample_indices, sequence_matrix, sequence_lengths, labels):
    sequences = subsample_sequence_matrix(sample_indices, sequence_matrix, sequence_lengths)
    subsampled_lengths = sequence_lengths[sample_indices]
    subsampled_labels = labels[sample_indices]
    return sequences, subsampled_lengths, subsampled_labels


def split_list(l, k):
    n = len(l) / k
    r = len(l) % k
    ret = [n+1] * (r) + [n] * (k-r)
    return [l[sum(ret[0:i]):sum(ret[0:i])+ret[i]] for i in xrange(k)]


# open and fetch nc file list inputs variable and return also some other stuff
def fetch_nc_inputs(nc_files):
    file_list = [NetCDFFile(f, 'a') for f in nc_files]
    variable_list = [f.variables['inputs'] for f in file_list]
    inputs_list = [var.value for var in variable_list]
    return file_list, variable_list, inputs_list


def subtract_mean_divide_std_from_nc(file_list, variable_list, inputs_list):
    num_files = len(file_list)
    concated_input = np.vstack(inputs_list)
    nc_mean = np.mean(concated_input)
    nc_std = np.std(concated_input)

    del concated_input

    for var in variable_list:
        var[:] -= nc_mean
        var[:] /= nc_std

    for i in xrange(num_files):
        file_list[i].close()
        del inputs_list[i]

    return nc_mean, nc_std