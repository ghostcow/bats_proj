function [ wav, length ] = load_wav(wavname, wavpath)
%helper function to load wav from bats dataset.
% returns wav signal and length
fname = [wavname, '.WAV']; % 'T0026803_4_01-10-12_3' is a file name for example
fname = [wavpath, filesep, fname];
wav = audioread(fname);
length = size(wav,1);
end

