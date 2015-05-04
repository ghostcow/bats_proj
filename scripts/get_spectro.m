function spectro = get_spectro(wav, window_length, noverlap, nfft, sample_rate)
%% function that extracts log spectrogram from wav file    

spectro = spectrogram(wav, window_length, noverlap, nfft, sample_rate, 'yaxis');
spectro = abs(spectro);
spectro = flipud(spectro);
spectro = 10*log(spectro);