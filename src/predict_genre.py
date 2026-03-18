import os
import json
import numpy as np
import tensorflow.keras as keras
import librosa

# --- CONFIGURATION ---
MODEL_PATH = "pakistani_music_model.h5"
SAMPLE_RATE = 22050
TRACK_DURATION = 30 # We analyze 30 seconds of the song
SLICE_DURATION = 3  # We cut it into 3-second slices
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

# GENRE MAPPING (Must match the order from your training!)
# 0: Bhangra, 1: Ghazal, 2: HipHop, 3: Pop, 4: Qawwali
GENRES = ["Bhangra", "Ghazal", "HipHop", "Pop", "Qawwali"]

def process_input_song(file_path):
    """
    Takes a song path, cuts the middle 30 seconds, 
    chops it into 10 slices, and turns them into math (MFCCs).
    """
    try:
        # 1. Load the full audio
        # (suppress warnings for cleaner output)
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 2. Find the middle of the song (to avoid slow intros)
        song_length = len(audio)
        if song_length > SAMPLES_PER_TRACK:
            mid_point = song_length // 2
            start_sample = max(0, mid_point - (SAMPLES_PER_TRACK // 2))
            end_sample = start_sample + SAMPLES_PER_TRACK
            audio_segment = audio[start_sample:end_sample]
        else:
            # If song is shorter than 30s, just take what we have
            audio_segment = audio

        # 3. Chop into 3-second slices
        SAMPLES_PER_SLICE = SAMPLE_RATE * SLICE_DURATION
        num_slices = int(len(audio_segment) / SAMPLES_PER_SLICE)
        
        processed_slices = []
        
        for s in range(num_slices):
            start = s * SAMPLES_PER_SLICE
            end = start + SAMPLES_PER_SLICE
            slice_audio = audio_segment[start:end]
            
            # Extract MFCC (The "Fingerprint")
            mfcc = librosa.feature.mfcc(y=slice_audio, sr=SAMPLE_RATE, n_mfcc=13, n_fft=2048, hop_length=512)
            mfcc = mfcc.T # Transpose to match training shape
            
            # Ensure shape is correct (sometimes end slices are tiny)
            expected_shape = (130, 13) 
            if mfcc.shape == expected_shape:
                processed_slices.append(mfcc)
            
        return np.array(processed_slices)

    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_genre(file_path):
    # 1. Load the trained brain
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found. Did you run train.py?")
        return

    model = keras.models.load_model(MODEL_PATH)
    
    # 2. Process the song
    print(f"\nProcessing: {os.path.basename(file_path)}...")
    X = process_input_song(file_path)
    
    if X is None or len(X) == 0:
        print("Could not extract data from song.")
        return

    # 3. Add the 'channel' dimension for CNN (batch, 130, 13, 1)
    X = X[..., np.newaxis]

    # 4. Predict on ALL slices at once
    predictions = model.predict(X, verbose=0)
    
    # 5. Vote
    votes = np.argmax(predictions, axis=1)
    counts = np.bincount(votes, minlength=len(GENRES))
    
    # 6. Winner
    winner_index = np.argmax(counts)
    winner_genre = GENRES[winner_index]
    confidence = (counts[winner_index] / len(votes)) * 100
    
    print("\n-----------------------------")
    print(f"PREDICTION:  {winner_genre.upper()}")
    print(f"Confidence:  {confidence:.1f}%")
    print("-----------------------------")
    print(f"Vote Breakdown: {dict(zip(GENRES, counts))}")

if __name__ == "__main__":
    print("Model Loaded. Ready to predict.")
    while True:
        print("\n" + "="*30)
        user_input = input("Drag and drop a song file here (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break

        # --- SMART CLEANING (The Fix) ---
        # 1. Remove extra spaces
        song_path = user_input.strip()
        
        # 2. If it starts with '&', remove only that first character
        if song_path.startswith("&"):
            song_path = song_path[1:].strip()
            
        # 3. Remove single/double quotes from start and end ONLY
        song_path = song_path.strip("'").strip('"')
        # -------------------------------

        if os.path.exists(song_path):
            predict_genre(song_path)
        else:
            print(f"Error: File not found at: {song_path}")
            print("Check if the file name has special characters or try renaming it to something simple like 'test.mp3'")