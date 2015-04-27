from make_nc import *
import h5py
from nc_functions import *
from sklearn.cross_validation import KFold, train_test_split
import cPickle


nc_path = '/home/lioruzan/bats_proj/data/nc_test'
mat_file = '/home/lioruzan/bats_proj/data/spectrograms/500_250_concat.mat'
classification_task_num = 0  # first classif. task out of 7
kfold_num = 5


if __name__ == '__main__':

    ## bat .mat file for data
    mat = h5py.File(mat_file)

    # load sequences and length info
    sequence_matrix = mat['concat_spectro'].value
    sequence_lengths = mat['lengths'].value

    # reshape lengths to vector
    sequence_lengths = sequence_lengths.astype(np.int32)
    sequence_lengths = sequence_lengths.reshape((sequence_lengths.shape[0],))

    # load labels
    label_matrix = mat['sequence_labels'].value
    label_maps = process_label_matrix(label_matrix)
    num_labels = len(label_maps[classification_task_num])
    labels = label_matrix[:, classification_task_num]

    mat.close()

    # split to train and test, load if split exists
    num_sequences = len(sequence_lengths)
    splits_path = '/home/lioruzan/bats_proj/metadata/splits.pkl'
    if os.path.isfile(splits_path):
        f = open(splits_path)
        d = cPickle.load(f)
        train_indices, test_indices = d['train_split'], d['test_split']
        f.close()
    else:
        train_indices, test_indices = train_test_split(range(num_sequences), test_size=0.2)
        # save splits for future reference
        save_file = open(splits_path, 'w')
        cPickle.dump({'train_split': train_indices, 'test_split': test_indices}, save_file)
        save_file.close()

    # calculate mean/std and normalize
    train_set = subsample_sequence_matrix(train_indices, sequence_matrix, sequence_lengths)
    test_set = subsample_sequence_matrix(test_indices, sequence_matrix, sequence_lengths)

    train_mean = np.mean(train_set)
    train_std = np.std(train_set)
    del train_set
    del test_set

    sequence_matrix -= train_mean
    sequence_matrix /= train_std

    # split randomized train indices to k sets for cross-validation
    cv_sets = split_list(train_indices, kfold_num)

    ## make .nc files out of all train/val/test splits
    # write out test
    nc_name = 'test.nc'
    sample_indices = test_indices
    test_sequences, test_lengths, test_labels = subsample_sequences(
        sample_indices, sequence_matrix, sequence_lengths, labels)
    make_nc(test_sequences, test_labels, num_labels, test_lengths, nc_name, nc_path)

    # write out all cross validation sets
    for j in xrange(kfold_num):
        nc_name = 'train_{}fold_{:02}_{:02}.nc'.format(kfold_num, classification_task_num, j)  # example: train_5fold_01_00.nc
        sample_indices = cv_sets[j]
        train_sequences, train_lengths, train_labels = subsample_sequences(
            sample_indices, sequence_matrix, sequence_lengths, labels)
        make_nc(train_sequences, train_labels, num_labels, train_lengths, nc_name, nc_path)
