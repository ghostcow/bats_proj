from make_nc import *
from nc_functions import *


nc_path = '/home/lioruzan/bats_proj/data/nc'
mat_file = '/home/lioruzan/bats_proj/data/spectrograms/500_250_concat.mat'
label_matrix = load_label_matrix(mat_file)
label_maps = process_label_matrix(label_matrix)


if __name__ == '__main__':

    ## for debugging purposes ONLY:
    col = 0
    num_cols = label_matrix.shape[1]
    for col in xrange(num_cols):
        labels = label_matrix[:, col]
        nc_name = str(col)
        make_nc(mat_file, labels, nc_name, nc_path)

    make_nc(mat_file, labels, nc_name, nc_path)
    ## actual code
    # num_cols = label_matrix.shape[1]
    # for col in xrange(num_cols):
    #     labels = label_matrix[:, col]
    #     nc_name = str(col)
    #     make_nc(mat_file, labels, nc_name, nc_path)

