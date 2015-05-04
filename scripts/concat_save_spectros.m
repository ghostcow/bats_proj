
p = '/home/lioruzan/bats_proj/data/spectrograms/500_250';
nsamples = single(10266);

% calculate lengths of all spectrograms
lengths = zeros(1, nsamples, 'single');

for i=1:nsamples
    S = load([p, filesep, num2str(i), '.mat'], 'spectro');
    lengths(i) = size(S.spectro,2);
    % length = length + size(S.spectro,2);
    clear S;
end

total_length = sum(lengths);
total_length = single(total_length);
concat_spectro = zeros(257, total_length, 'single');


% concatenate all spectrograms and dump to hdd
s = 1;
e = lengths(1);
for i=1:nsamples
    
    S = load([p, filesep, num2str(i), '.mat'], 'spectro');
    A = single(S.spectro);
    concat_spectro(:,s:e) = A;
    
    % set next spectro margins
    if i ~= nsamples
        s = e + 1;
        e = e + lengths(i+1);
    end
    
    % dump old memory
    clear A;
    clear S;
end

num_seq = nsamples;

% get labels
load('/home/lioruzan/bats_proj/metadata/bats.mat', 'seqAnnotation')
label_columns = [4 5 6 7 8 9 11];
sequence_labels = seqAnnotation(:,label_columns)';
save('/home/lioruzan/bats_proj/data/spectrograms/500_250_concat.mat', 'concat_spectro', 'lengths', 'total_length', 'num_seq', 'sequence_labels', '-v7.3');