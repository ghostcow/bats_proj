
base_dir = '/home/lioruzan/bats_proj';
metadata_dir = [base_dir, filesep, 'metadata'];
data_dir = [base_dir, filesep, 'data'];
spectrograms_dir = [data_dir, filesep, 'spectrograms'];

load([metadata_dir, filesep, 'bats.mat'], 'seqIdx')
load([metadata_dir, filesep, 'bats.mat'], 'FileNames')
load([metadata_dir, filesep, 'bats.mat'], 'Duration')

% parameters
wavpath = [data_dir, filesep, 'adult_calls'];
num_sequences = size(seqIdx,2);
sample_rate = 250E3;

window_length = 500;
overlap_length = 250;
num_fft_points = [];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% other window parameter options
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% window_length = 625;
% overlap_length = 312;

% window_length = 2500;
% overlap_length = 1250;

% window_length = 1250;
% overlap_length = 625;

% window_length = 5000;
% overlap_length = 2500;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

window_length_specific_dir = [spectrograms_dir, filesep, num2str(window_length), '_', num2str(overlap_length)];
mkdir(window_length_specific_dir)

for i=1:num_sequences
    
    % calculate length of sequence
    seq_id = seqIdx{i};
    length = 0;
    for j = seq_id
        
        fname = FileNames{j};
        cur_length = get_wav_length(fname,wavpath);
        length = length + cur_length;
    end
    
    % allocate memory + concat wav signals
    length = floor(length);
    sequence = zeros(length,1);
    current_position = 1;
    for j = seq_id
        
        fname = FileNames{j};
        cur_length = get_wav_length(fname,wavpath);
        cur_wav = load_wav(fname, wavpath)';
        next_position = floor(current_position+cur_length-1);
        sequence(current_position:next_position) = cur_wav;
        current_position = current_position + cur_length;
    end
    
    spectro = get_spectro(sequence,window_length,overlap_length,num_fft_points,sample_rate);

    save([window_length_specific_dir, filesep, num2str(i)], 'spectro');
    clear sequence; clear spectrogram;
end