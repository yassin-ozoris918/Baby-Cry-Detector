import os
import numpy as np
from collections import Counter
from configs.config import BABY_DATASET, CRY_CATEGORIES, OUTPUT_SPEECH_DATASET
from utils.features_extraction import extract_feature_vector




def dataset_builder():
    features, labels = [], []

    # Baby Cry Dataset

    for dir in os.listdir(BABY_DATASET):
        dir_path = os.path.join(BABY_DATASET, dir)

        if not os.path.isdir(dir_path):
            continue

        label = "cry" if dir in CRY_CATEGORIES else "not_cry"

        for file in os.listdir(dir_path):
            if file.endswith(".wav"):
                file_path = os.path.join(dir_path, file)
                features.append(extract_feature_vector(file_path))
                labels.append(label)



    # Speech Dataset

    for spk in os.listdir(OUTPUT_SPEECH_DATASET):
        spk_path = os.path.join(OUTPUT_SPEECH_DATASET, spk)
        for file in os.listdir(spk_path):
            if file.endswith(".wav"):
                file_path = os.path.join(spk_path, file)
                features.append(extract_feature_vector(file_path))
                labels.append("not_cry")


    print(f"Dataset Size => {len(features)}")
    print(f"Class Countes => {Counter(labels)}")

    return np.array(features), np.array(labels)