__author__ = 'lioruzan'
import numpy as np
import csv
from Scientific.IO.NetCDF import *
from scipy.stats import mode
from sklearn.metrics import accuracy_score, confusion_matrix
import tsne
import pylab as Plot


# @pre: csv are lists of activations of sequences, in ascending order, in CURRENNT output format.
def parse_csv(csv_file, num_activation_units):
    activations = []
    with open(csv_file, 'r') as f:
        for activation in csv.reader(f, delimiter=';'):
            activation_shape = (len(activation) / num_activation_units, num_activation_units)
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


# to use only on softmax probability type activations
def calculate_ff_accuracy(csv_file, activation_size, nc_file):
    global activations, timestep_tru_labels, sequence_lengths, timestep_prediction_labels, flat_timestep_prediction_labels, timestep_accuracy, sequence_ground_truth, sequence_predictions, sequence_accuracy

    # activations of all timesteps
    activations = parse_csv(csv_file, activation_size)

    # labels of sequences of validation set
    timestep_tru_labels, sequence_lengths = get_labels_lengths_from_nc(nc_file)

    # get timestep labels from activations
    timestep_prediction_labels = get_feedforward_labels(activations)

    # calculate true accuracy - relevent only for class probability activations
    # flatten prediction labels to calculate accuracy
    flat_timestep_prediction_labels = flatten_list(timestep_prediction_labels)
    timestep_accuracy = accuracy_score(timestep_tru_labels, flat_timestep_prediction_labels)
     
    sequence_ground_truth = restore_seq_label_list(timestep_tru_labels, sequence_lengths)
    sequence_predictions = seq_labels_by_majority(timestep_prediction_labels)
    sequence_accuracy = accuracy_score(sequence_ground_truth, sequence_predictions)

    print('timestep accuracy: ' + str(timestep_accuracy))
    print('sequence accuracy: ' + str(sequence_accuracy))
    print('confusion matrix:')
    print(confusion_matrix(sequence_ground_truth, sequence_predictions))


def t_sne(activation_size, csv_file, nc_file):
    global timestep_activations, seq_activations, timestep_tru_labels, sequence_lengths, seq_labels, Y
    ################################################################################################
    # t-SNE code
    ################################################################################################
    '''
        tasks:
        V 1. write code to average activations of a sequence for seq representations
        V 2. run t-SNE on softmax activations
        V 3. create bogus nc files to use to extract last layer activations

        other ideas:
        1. take last activation label for improved accuracy + better sequence representation?
        3. use mode of 2nd half of sequence for improved accuracy
        4. use rnnlib to classify entire sequences instead of currennt.
        '''
    # activations
    timestep_activations = parse_csv(csv_file, activation_size)
    seq_activations = np.array(average_seq_activations(timestep_activations), dtype=np.float64)
    # labels
    timestep_tru_labels, sequence_lengths = get_labels_lengths_from_nc(nc_file)
    seq_labels = restore_seq_label_list(timestep_tru_labels, sequence_lengths)
    seq_labels = np.array(seq_labels)
    Y = tsne.tsne(seq_activations)
    Plot.scatter(Y[:, 0], Y[:, 1], 20, seq_labels)
    Plot.show()
