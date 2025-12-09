import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import freqz
from configs.config import LOWCUT, HIGHCUT, SAMPLE_RATE, N_TAPS
from utils.audio_utils import apply_filter, butter_bandpass, FIR_bandpass


def make_1d_audio(audio):
    """
    Ensure the audio is a 1D numpy array.
    Handles stereo or nested lists safely.
    """
    # If it's a list of arrays, flatten it first
    if isinstance(audio, list):
        audio = np.concatenate(audio)

    # If it's 2D (stereo), take the first channel
    if len(audio.shape) > 1:
        audio = audio[:, 0]

    # Ensure 1D float array
    audio = np.asarray(audio, dtype=float).flatten()
    return audio



def plot_freq_response(b, a, label="Filter"):
    omiga, h = freqz(b, a, worN=2048)
    freq = (omiga / (2 * np.pi)) * SAMPLE_RATE


    plt.plot(freq, 20 * np.log10(np.abs(h)), label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Frequency Response")
    plt.grid(True)
    


def compare_filters(audio_path):
    # Load audio
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio  = make_1d_audio(audio)



    # Apply FIR Filter
    y_fir = apply_filter(audio, filter_type="FIR")
    b_fir = FIR_bandpass()
    a_fir = 1.0

    # Apply IIR Filter
    y_iir = apply_filter(audio, filter_type="IIR")
    b_iir, a_iir = butter_bandpass()


    # ----------------------------------------------------1) Plot Frequency Response -----------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plot_freq_response(b_fir, a_fir, label="FIR Filter")
    plot_freq_response(b_iir, a_iir, label="IIR Filter")
    plt.legend()
    plt.show()

    # ----------------------------------------------------2) Time Domain Comparison -----------------------------------------------------------

    plt.figure(figsize=(12, 4))
    plt.plot(audio, label="Original", alpha=0.5)
    plt.plot(y_iir, label="IIR Output")
    plt.plot(y_fir, label="FIR Output")
    plt.title("Time Domain (zoomed)")
    plt.xlim(0, 2000)
    plt.legend()
    plt.grid()
    plt.show()
    # ----------------------------------------------------3) Spectrograms --------------------------------------------------------------------

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].specgram(audio, Fs=SAMPLE_RATE)
    ax[0].set_title("Original")

    ax[1].specgram(y_iir, Fs=SAMPLE_RATE)
    ax[1].set_title("IIR Output")

    ax[2].specgram(y_fir, Fs=SAMPLE_RATE)
    ax[2].set_title("FIR Output")

    plt.tight_layout()
    plt.show()




if __name__ == "__main__" :

    audio_path = "plotting_audio/1.wav"
    compare_filters(audio_path)