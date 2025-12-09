# real_time_cry_detector.py

import sounddevice as sd
import numpy as np
import joblib
import librosa
import time
import os
from twilio.rest import Client
from configs.config import SAMPLE_RATE, FRAME_LENGTH, HOP_LENGTH, N_MFCC, MODEL_OUT_DIR
from utils.audio_utils import apply_filter
# ------------------------------------------------------------Load trained model + encoder + scaler---------------------------------------------
svm_model = joblib.load(os.path.join(MODEL_OUT_DIR, 'svm_cry_detector.pkl'))
scaler = joblib.load(os.path.join(MODEL_OUT_DIR, 'feature_scaler.pkl'))
label_encoder = joblib.load(os.path.join(MODEL_OUT_DIR, 'label_encoder.pkl'))

print("Model, Scaler, Encoder Loaded.")



# ------------------------------------------------------------Extract feature vector from audio-------------------------------------------
def extract_features_from_audio(audio):
    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-6)

    # Bandpass IIR filter
    y = apply_filter(audio)

    # MFCC + delta
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, hop_length =HOP_LENGTH, win_length=FRAME_LENGTH, n_mfcc=N_MFCC)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Statistics
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)

    feature_vector = np.concatenate([mfcc_mean, mfcc_delta_mean, mfcc_std, mfcc_delta_std])
    return feature_vector
# -------------------------------------------------------------WhatsApp Sending Function-----------------------------------------------------------------
TWILIO_SID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_AUTH = "xxxxxxxxxxxxxxxxxxxxxxxxx"
WHATS_APP_TO = "xxxxxxxxxxxxxxxxxxxxx"
WHATS_APP_FROM = "xxxxxxxxxxxxxxxxx"

client = Client(TWILIO_SID, TWILIO_AUTH)
baby_name = "Seif"

def send_whatsapp_message(baby_name="your baby"):
    message = f"🚨{baby_name} is crying , please check up🚨"
    try:
        message = client.messages.create(body=message, from_=WHATS_APP_FROM, to=WHATS_APP_TO)
        print(f"WhatsApp sent => {message.sid}")
    except Exception as e:
        print(f"Error sending whatsApp => {e}")




# ---------------------------------------------------------------Real-time loop-----------------------------------------------------------------

print("Baby cry detector is running ...")
print("Listening...")

duration = 3  # seconds
history_len = 5
history = []

cooldown = 0  # cooldown counter in seconds
cooldown_duration = 2  # seconds after a detected cry

try:
    while True:
        # 1) record audio from mic 
        audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        y = audio.flatten()

        # RMS energy
        rms = np.sqrt(np.mean(y**2))

        # Skip silent audio
        if rms < 0.01:
            print("No cry (too quiet)")
            history = []  # reset history
            time.sleep(0.1)
            continue

        # 2) extract features 
        feature_vector = extract_features_from_audio(y)

        # 3) scale features
        scaled_features = scaler.transform([feature_vector])

        # 4) predict
        prediction = svm_model.predict(scaled_features)[0]
        history.append(prediction)
        if len(history) > history_len:
            history.pop(0)

        # 5) Smooth decision with history + cooldown
        cry_votes = history.count(1)
        if cry_votes >= history_len // 2 + 1 and cooldown <= 0:
            print("🚨 CRY DETECTED! 🚨")
            send_whatsapp_message(baby_name)
            cooldown = cooldown_duration  # start cooldown
            history = []  # reset history after detection
        else:
            print("No cry.")

        # 6) Update cooldown
        if cooldown > 0:
            cooldown -= duration

        # Debug info
        print(f"RMS: {rms:.3f}, Feature mean: {np.mean(feature_vector):.3f}, Raw pred: {prediction}")

        time.sleep(0.1)

except KeyboardInterrupt:
    print("\nStopping detector...")
