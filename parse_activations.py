__author__ = 'lioruzan'
import numpy as np
from activations_parsing_functions import *


# parameters
num_classes = 10
activation_size = 10
csv_file = '/home/lioruzan/bats_proj/rnn/500_250/task00_00/ff_softmax_output_test032.csv'
nc_file = '/home/lioruzan/bats_proj/data/nc/test_00.nc'


if __name__ == '__main__':

    ## calculate softmax probability classification accuracy
    calculate_ff_accuracy(csv_file, activation_size, nc_file)

    ## run t-SNE
    #t_sne(activation_size, csv_file, nc_file)
