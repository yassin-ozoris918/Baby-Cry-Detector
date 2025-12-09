# Config.py      =====> contains all settings 

SAMPLE_RATE = 16000      # 16 KHz
FRAME_LENGTH = 1024
HOP_LENGTH = FRAME_LENGTH // 2
N_MFCC = 13
LOWCUT = 300      # 300 Hz
HIGHCUT = 3000    # 3 KHz
N_TAPS = 101     # For FIR Filter


# Dataset Paths

LIBRISPEECH_PATH = "datasets/librispeech_raw/train-clean-100"
OUTPUT_SPEECH_DATASET = "datasets/speech_wav"
BABY_DATASET = "datasets/baby_dataset"


MODEL_OUT_DIR = "model_artifacts"

N_SPEAKERS = 100

CRY_CATEGORIES = ["belly pain", "burping", "cold_hot", "discomfort", "hungry", "tired"]
