import os, random, librosa
import soundfile as sf
from configs.config import LIBRISPEECH_PATH, OUTPUT_SPEECH_DATASET, SAMPLE_RATE, N_SPEAKERS




os.makedirs(OUTPUT_SPEECH_DATASET, exist_ok=True) 

all_speakers = [d for d in os.listdir(LIBRISPEECH_PATH)]

print("Total available speakers:", len(all_speakers))


selected_speakers = random.sample(all_speakers, N_SPEAKERS)

print("converting FLAC to WAV...")

for spk in selected_speakers:

    spk_path = os.path.join(LIBRISPEECH_PATH, spk)

    spk_out_path = os.path.join(OUTPUT_SPEECH_DATASET, spk)
    os.makedirs(spk_out_path, exist_ok=True)


    for ch in os.listdir(spk_path):
        ch_path = os.path.join(spk_path, ch)

        for file in os.listdir(ch_path):
            if file.endswith(".flac"):
                src = os.path.join(ch_path, file)

                dst = os.path.join(spk_out_path, file.replace(".flac", ".wav"))

                audio, _ = librosa.load(src, sr=SAMPLE_RATE)

                sf.write(dst, audio, SAMPLE_RATE)

print("Done.")


