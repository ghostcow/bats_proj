from make_nc import *
from nc_functions import *


nc_path = '/home/lioruzan/bats_proj/data/nc'
mat_file = '/home/lioruzan/bats_proj/data/spectrograms/500_250_concat.mat'

## bat .mat file for data
mat = h5py.File(mat_file)

# load sequences and length info
concat_spectro = mat['concat_spectro'].value
lengths = mat['lengths'].value
# reshape lengths to vector
lengths = lengths.astype(np.int32)
lengths = lengths.reshape((lengths.shape[0],))

# load labels
label_matrix = mat['sequence_labels'].value
label_maps = process_label_matrix(label_matrix)

mat.close()


# from gutted make_nc function
_make_nc(concat_spectro, labels, lengths, nc_name, nc_path)


if __name__ == '__main__':

    ## for debugging purposes ONLY:
    col = 0
    num_cols = label_matrix.shape[1]
    labels = label_matrix[:, col]
    nc_name = str(col) + '.nc'
    make_nc(mat_file, labels, nc_name, nc_path)

    ## actual code
    # num_cols = label_matrix.shape[1]
    # for col in xrange(num_cols):
    #     labels = label_matrix[:, col]
    #     nc_name = str(col)
    #     make_nc(mat_file, labels, nc_name, nc_path)

