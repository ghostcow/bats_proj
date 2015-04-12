__author__ = 'lioruzan'
import h5py
import numpy as np


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


# also takes only vector one-dimensional sequence_lengths, labels.
def subsample_sequences(sample_indices, sequence_matrix, sequence_lengths, labels):
    sequences = [get_sequence_from_matrix(i, sequence_matrix, sequence_lengths) for i in sample_indices]
    sequences = np.vstack(sequences)
    subsampled_lengths = sequence_lengths[sample_indices]
    subsampled_labels = labels[sample_indices]
    return sequences, subsampled_lengths, subsampled_labels
