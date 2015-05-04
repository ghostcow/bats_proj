function [ length ] = spectro_length(wavname, wavpath, window, noverlap)
%SPECTRO_LENGTH calculates length of spectrogram from wav
[~, wav_length] = load_wav(wavname,wavpath);
stride = window - noverlap;
length = (wav_length - noverlap) / stride;
length = floor( length );
end

