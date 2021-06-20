import numpy as np
import librosa



def wav2mfcc(wave, sr, max_pad_len=20):
    
    hop_length = 128
    wave_len_limit = 4096
    input_wave_len = len(wave)
    if input_wave_len > wave_len_limit:
        wave  = wave[0:4096]
    elif input_wave_len < wave_len_limit:

        padding = wave_len_limit - input_wave_len

        wave = np.pad(wave,(0,padding))


    # wave = wave[::3] # why decimation?
    mfcc = librosa.feature.mfcc(wave, sr, hop_length = hop_length)

    return mfcc
