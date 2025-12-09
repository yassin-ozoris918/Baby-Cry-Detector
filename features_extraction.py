import numpy as np
from utils.audio_utils import load_audio
import librosa
from configs.config import SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH, N_MFCC



def extract_feature_vector(path):
    y = load_audio(path)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, hop_length = HOP_LENGTH, win_length = FRAME_LENGTH, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)


    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)

    return np.concatenate([mfcc_mean, mfcc_delta_mean, mfcc_std, mfcc_delta_std])


    