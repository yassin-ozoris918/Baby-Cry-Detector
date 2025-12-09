import numpy as np
import librosa
from scipy.signal import butter, lfilter, firwin
from configs.config import SAMPLE_RATE, LOWCUT, HIGHCUT, N_TAPS

def butter_bandpass(lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLE_RATE, order=5):
    nyq = 0.5 * fs

    norm_lowcut = lowcut / nyq
    norm_highcut = highcut / nyq
    return butter(order, [norm_lowcut, norm_highcut], btype='band')


def FIR_bandpass(lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLE_RATE, n_taps=N_TAPS):
    return np.array(firwin(n_taps, [lowcut, highcut], pass_zero=False, fs=fs))



def apply_filter(signal, filter_type="IIR"):
    if filter_type == "FIR":
        b = FIR_bandpass()
        a = 1.0
    else:
        b, a = butter_bandpass()
    return lfilter(b, a, signal)


def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    y = apply_filter(y)
    return y