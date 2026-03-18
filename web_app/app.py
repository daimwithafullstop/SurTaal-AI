import os
from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow.keras as keras
import librosa

app = Flask(__name__)

MODEL_PATH = "models/pakistani_music_model.h5"
GENRES = ["Bhangra", "Ghazal", "HipHop", "Pop", "Qawwali"]
SAMPLE_RATE = 22050
TRACK_DURATION = 30
SLICE_DURATION = 3
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# load model
model = keras.models.load_model(MODEL_PATH)
print("--> SurTaal AI Engine Loaded!")

def process_audio(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Cut Middle 30s
        if len(audio) > SAMPLES_PER_TRACK:
            mid = len(audio) // 2
            start = max(0, mid - (SAMPLES_PER_TRACK // 2))
            end = start + SAMPLES_PER_TRACK
            audio = audio[start:end]

        SAMPLES_PER_SLICE = SAMPLE_RATE * SLICE_DURATION
        num_slices = int(len(audio) / SAMPLES_PER_SLICE)
        processed_slices = []
        for s in range(num_slices):
            start = s * SAMPLES_PER_SLICE
            end = start + SAMPLES_PER_SLICE
            slice_audio = audio[start:end]
            
            mfcc = librosa.feature.mfcc(y=slice_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc = mfcc.T
            if mfcc.shape == (130, 13):
                processed_slices.append(mfcc)
        
        return np.array(processed_slices)
    except Exception as e:
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file:
        file_path = "data/temp_upload.mp3"
        file.save(file_path)

        X = process_audio(file_path)
        
        if X is not None and len(X) > 0:
            X = X[..., np.newaxis]
            predictions = model.predict(X, verbose=0)
            
            votes = np.argmax(predictions, axis=1)
            counts = np.bincount(votes, minlength=len(GENRES))
            winner_idx = np.argmax(counts)
            winner_genre = GENRES[winner_idx]
            confidence = (counts[winner_idx] / len(votes)) * 100
            
            os.remove(file_path)

            return jsonify({
                'genre': winner_genre.upper(),
                'confidence': f"{confidence:.1f}%",
                'scores': counts.tolist(),
                'labels': GENRES
            })
        else:
            return jsonify({'error': 'Could not process audio'})

if __name__ == '__main__':
    app.run(debug=False, threaded=False)