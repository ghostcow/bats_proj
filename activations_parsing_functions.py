__author__ = 'lioruzan'
import numpy as np
import csv
from Scientific.IO.NetCDF import *
from scipy.stats import mode


# @pre: csv are lists of activations of sequences, in ascending order, in CURRENNT output format.
def parse_csv(csv_file, num_classes):
    activations = []
    with open(csv_file, 'r') as f:
        for activation in csv.reader(f, delimiter=';'):
            activation_shape = (len(activation) / num_classes, num_classes)
            activation = activation[1:]  # remove tag, it's is sequential from 0, so not necessary when appending
            activation = np.array(activation, 'float32').reshape(activation_shape)  # make ndarray
            activations.append(activation)

    return activations


def get_labels_lengths_from_nc(nc_fname):
    nc_file = NetCDFFile(nc_fname, 'r')
    labels = nc_file.variables['targetClasses'].getValue()
    lengths = nc_file.variables['seqLengths'].getValue()
    nc_file.close()
    return labels, lengths


def get_seq_label(i, labels, lengths):
    tgt_label_index = sum(lengths[0:i])
    return labels[tgt_label_index]


def get_feedforward_labels(activations):
    num_sequences = len(activations)
    labels = [[] for i in xrange(num_sequences)]
    for i in xrange(num_sequences):

        n_timesteps = activations[i].shape[0]
        for j in xrange(n_timesteps):

            label = np.argmax(activations[i][j])
            labels[i].append(label)
    return labels


# restores original sequence classification labels from expanded timestep label list
def restore_seq_label_list(timestep_labels, sequence_lengths):
    l = []
    num_sequences = len(sequence_lengths)
    for i in xrange(num_sequences):
        l.append(get_seq_label(i, timestep_labels, sequence_lengths))
    return l


def seq_labels_by_majority(timestep_labels):

    num_sequences = len(timestep_labels)
    seq_labels = np.zeros(num_sequences, 'int32')
    for i in xrange(num_sequences):

        current_label = mode(timestep_labels[i])
        current_label = current_label[0][0]
        current_label = np.int32(current_label)
        seq_labels[i] = current_label
    return seq_labels


# flatten 2-D list of lists
def flatten_list(l):
    return [val for sublist in l for val in sublist]


# accepts list of timestep activations of sequences
def average_seq_activations(activations):
    return [np.average(l, axis=0) for l in activations]

