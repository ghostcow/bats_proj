function [ length ] = get_wav_length(wavname, wavpath)
%returns wav signal length
fname = [wavname, '.WAV']; % 'T0026803_4_01-10-12_3' is a file name for example
fname = [wavpath, filesep, fname];
wav = audioread(fname);
length = size(wav,1);
end