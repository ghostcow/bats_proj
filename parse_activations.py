__author__ = 'lioruzan'
import numpy as np
from activations_parsing_functions import *




# parameters
num_classes = 10
activation_size = 128
#activation_size = 10

csv_file = '/home/lioruzan/bats_proj/rnn/500_250/task00_00/ff_output_test029.csv'
#csv_file = '/home/lioruzan/bats_proj/rnn/500_250/task00_00/ff_softmax_output_test029.csv'

nc_file = '/home/lioruzan/bats_proj/data/nc/test.nc'
#validation_part = 2


if __name__ == '__main__':

    ## calculate softmax probability classification accuracy
    # calculate_ff_accuracy(csv_file, activation_size, nc_file)

    ## run t-SNE
    t_sne(activation_size, csv_file, nc_file)