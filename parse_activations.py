__author__ = 'lioruzan'
import numpy as np
from activations_parsing_functions import *
from sklearn.metrics import accuracy_score
import tsne
import pylab as Plot



# parameters
num_classes = 10
csv_file = '/home/lioruzan/bats_proj/rnn/500_250/task00_00/test001_22.52.percent.error_ff.csv'
nc_file = '/home/lioruzan/bats_proj/data/nc/train_00_02.nc'
validation_part = 2


if __name__ == '__main__':

    # # activations of all timesteps
    # activations = parse_csv(csv_file, num_classes)
    #
    # # labels of sequences of validation set train_00_02 for starters
    # timestep_tru_labels, sequence_lengths = get_labels_lengths_from_nc(nc_file)
    #
    # # get timestep labels from activations
    # timestep_prediction_labels = get_feedforward_labels(activations)
    #
    # # calculate true accuracy
    # # flatten prediction labels to calculate accuracy
    # flat_timestep_prediction_labels = flatten_list(timestep_prediction_labels)
    # timestep_accuracy = accuracy_score(timestep_tru_labels, flat_timestep_prediction_labels)
    #
    # sequence_ground_truth = restore_seq_label_list(timestep_tru_labels, sequence_lengths)
    # sequence_predictions = seq_labels_by_majority(timestep_prediction_labels)
    # sequence_accuracy = accuracy_score(sequence_ground_truth, sequence_predictions)
    #
    # print('timestep accuracy: ' + str(timestep_accuracy))
    # print('sequence accuracy: ' + str(sequence_accuracy))


    ################################################################################################
    # t-SNE code
    ################################################################################################

    '''
    tasks:
    V 1. write code to average activations of a sequence for seq representations
    2. run t-SNE on softmax activations
    3. create bogus nc files to use to extract fc activations

    other ideas:
    1. write code to take last activation as label for accuracy + sequence representation
    2.a addd FC layer before softmax classificatin,
    2.b retrieve FC layer activations to use instead of softmax features
    3. for accuracy- use mode of 2nd half of sequence
    4. use rnnlib to classify entire sequences with CTC loss instead of currennt.
    5. expreiment with removing tanh activation functions from c
    '''

    # activations
    timestep_activations = parse_csv(csv_file, num_classes)
    seq_activations = np.array(average_seq_activations(timestep_activations), dtype=np.float64)
    # labels
    timestep_tru_labels, sequence_lengths = get_labels_lengths_from_nc(nc_file)
    seq_labels = restore_seq_label_list(timestep_tru_labels, sequence_lengths)
    seq_labels = np.array(seq_labels)

    Y = tsne.tsne(seq_activations)
    Plot.scatter(Y[:,0], Y[:,1], 20, seq_labels)
    Plot.show()