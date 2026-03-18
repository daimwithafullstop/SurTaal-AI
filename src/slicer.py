import os
from pydub import AudioSegment
from pydub.utils import make_chunks

# --- CONFIGURATION ---
# Folder structure: Pakistani_Music_Project -> Raw_Songs -> Qawwali
INPUT_ROOT = "Raw_Songs"
OUTPUT_ROOT = "Processed_Dataset"
CHUNK_LENGTH_MS = 3000  # 3 seconds

def process_audio():
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # Walk through all folders in Raw_Songs
    for genre in os.listdir(INPUT_ROOT):
        genre_path = os.path.join(INPUT_ROOT, genre)
        
        # Only process if it is a folder (like 'Qawwali')
        if os.path.isdir(genre_path):
            print(f"--> Found Genre: {genre}")
            
            # Create output folder for this genre
            save_path = os.path.join(OUTPUT_ROOT, genre)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            files = [f for f in os.listdir(genre_path) if f.endswith(".mp3")]
            print(f"    Processing {len(files)} songs...")

            for song_file in files:
                try:
                    full_path = os.path.join(genre_path, song_file)
                    audio = AudioSegment.from_mp3(full_path)
                    
                    # LOGIC:
                    # Qawwalis are long, so we take more slices.
                    # Others are short, so we take fewer.
                    if genre.lower() == "qawwali":
                        # Start at 2 minutes to skip slow intros
                        start_ms = 120 * 1000 
                        # Take 2 minutes of audio (plenty of data!)
                        duration_ms = 120 * 1000 
                    else:
                        # For Pop/HipHop, start at 45 seconds
                        start_ms = 45 * 1000
                        # Take 30 seconds of audio
                        duration_ms = 30 * 1000

                    # Cut the "Meat" of the song
                    meat = audio[start_ms : start_ms + duration_ms]
                    
                    # Slice into 3-second chunks
                    chunks = make_chunks(meat, CHUNK_LENGTH_MS)

                    # Save chunks
                    clean_name = song_file.replace(" ", "_").replace(".mp3", "")
                    for i, chunk in enumerate(chunks):
                        # Only save if the chunk is actually 3 seconds (ignore tiny end bits)
                        if len(chunk) == CHUNK_LENGTH_MS:
                            chunk_name = f"{genre}.{clean_name}.{i}.wav"
                            chunk.export(os.path.join(save_path, chunk_name), format="wav")
                            
                except Exception as e:
                    print(f"    Error on {song_file}: {e}")

    print("\nDONE! Check the 'Processed_Dataset' folder.")

if __name__ == "__main__":
    process_audio()