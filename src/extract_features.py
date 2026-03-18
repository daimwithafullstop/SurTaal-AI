import json
import os
import math
import librosa
import numpy as np

DATASET_PATH = "Processed_Dataset"
JSON_PATH = "data/data.json"
SAMPLE_RATE = 22050
TRACK_DURATION = 3 
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def save_mfcc(dataset_path, json_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    """Extracts MFCCs from music dataset and saves them into a json file"""

    
    data = {
        "mapping": [], 
        "labels": [],   
        "mfcc": []    
    }

    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    print("--> Starting Feature Extraction... (This might take a few minutes)")

    # Loop through all genre folders
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        if dirpath is not dataset_path:

            # Save genre label 
            semantic_label = dirpath.split("\\")[-1] 
            data["mapping"].append(semantic_label)
            print(f"\nProcessing Genre: {semantic_label}")

            for f in filenames:
                file_path = os.path.join(dirpath, f)
                
                try:
                    signal, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

                    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc = mfcc.T  

                    if len(mfcc) == num_mfcc_vectors_per_segment:
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i-1)
                        
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
        
    print(f"\nSUCCESS! All audio converted to math. Saved in '{json_path}'")

if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1)