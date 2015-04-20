__author__ = 'lioruzan'
import numpy as np
from activations_parsing_functions import *
from sklearn.metrics import accuracy_score


# parameters
num_classes = 10
nc_file = '/home/lioruzan/bats_proj/data/nc/train_00_02.nc'
validation_part = 2


if __name__ == '__main__':

    # activations of all timesteps
    csv_file = '/home/lioruzan/bats_proj/rnn/500_250/task00_00/test001_22.52.percent.error_ff.csv'
    activations = parse_csv(csv_file, num_classes)

    # labels of sequences of validation set train_00_02 for starters
    timestep_tru_labels, sequence_lengths = get_labels_lengths_from_nc(nc_file)

    # get timestep labels from activations
    timestep_prediction_labels = get_feedforward_labels(activations)

    # calculate true accuracy
    # flatten prediction labels to calculate accuracy
    flat_timestep_prediction_labels = flatten_list(timestep_prediction_labels)
    timestep_accuracy = accuracy_score(timestep_tru_labels, flat_timestep_prediction_labels)

    sequence_ground_truth = restore_seq_label_list(timestep_tru_labels, sequence_lengths)
    sequence_predictions = seq_labels_by_majority(timestep_prediction_labels)
    sequence_accuracy = accuracy_score(sequence_ground_truth, sequence_predictions)

    print('timestep accuracy: ' + str(timestep_accuracy))
    print('sequence accuracy: ' + str(sequence_accuracy))


    ################################################################################################
    # t-SNE code
    ################################################################################################

    '''
    tasks:
    1. write code to average activations of a sequence for seq representations
    2. run t-SNE on softmwax activations

    other ideas:
    1. write code to take last activation as label for accuracy + sequence representation
    2.a addd FC layer before softmax classificatin,
    2.b retrieve FC layer activations to use instead of softmax features
    3. for accuracy- use mode of 2nd half of sequence


    # t-SNE sucka!!!!!!