# Baby Cry Detection System

## Overview
This project implements a real-time baby cry detection system using audio signal processing and machine learning. It includes DSP-based preprocessing, MFCC feature extraction, ML model training, and real-time inference.

## Features
- Real-time microphone audio processing  
- DSP preprocessing (RMS, ZCR, MFCC, Delta MFCC)  
- ML classification (KNN, SVM, Random Forest)  
- Smoothing and cooldown logic  
- Modular project pipeline

## Project Structure
```
project/
│
├── config/
│   └── config.py
│
├── dataset_prep/
│   ├── load_audio.py
│   ├── split_data.py
│   └── labeling.py
│
├── features/
│   ├── extract_mfcc.py
│   ├── extract_zcr.py
│   └── extract_rms.py
│
├── training/
│   ├── train_model.py
│   └── evaluate_model.py
│
├── realtime/
│   ├── stream_listener.py
│   ├── predict_frame.py
│   └── logic.py
│
└── app.py
```

## Installation
```
pip install numpy scipy librosa scikit-learn sounddevice python-dotenv
```

## Usage
```
python app.py
```

## Future Improvements
- Mobile app UI  
- Deep learning models  
- Edge deployment (ESP32, RPi, Edge TPU)

## License
Educational and research use only.
