import os
import sys
import time
import numpy as np
from joblib import Parallel, delayed
import librosa
import soundfile as sf
import pandas as pd

DATA_DIR_MP3 = "audiodata/mp3"
DATA_DIR_WAV = "audiodata/wav"

SR = 14000
BIRDS_DIR_LIST = os.listdir(DATA_DIR_MP3)


# Create directories for wav files
if not os.path.exists(DATA_DIR_WAV):
    os.makedirs(DATA_DIR_WAV)

for bird in BIRDS_DIR_LIST:
    data_dir_bird = f"{DATA_DIR_WAV}/{bird}"
    if not os.path.exists(data_dir_bird):
        os.makedirs(data_dir_bird)


def convert_mp3_to_wav(filename, birdName, sr):
    """
    Converts the mp3 file to wav, with a sample_rate = 14000Hz and a single channel.
    """
    bird_filename_mp3 = f"{DATA_DIR_MP3}/{birdName}/{filename}"
    bird_filename_wav = f"{DATA_DIR_WAV}/{birdName}/{filename[:-4]}.wav"

    if bird_filename_mp3[-4:] == '.mp3' and not os.path.isfile(bird_filename_wav):
        bird_array, _ = librosa.load(bird_filename_mp3, mono=True, sr=sr)
        sf.write(bird_filename_wav, bird_array, sr)


def conversion_function(birdslist, sr, do_print=False):
    """
    Converts all of the mp3 files
    """
    for birdName in birdslist:
        bird_dir = f"{DATA_DIR_MP3}/{birdName}"
        birdsaudiolist = [f for f in os.listdir(bird_dir) if os.path.isfile(os.path.join(bird_dir, f))]
        
        if do_print: start_time = time.time(); print(f"Converting {len(birdsaudiolist)} audio files of {birdName}")
           
        Parallel(n_jobs=-1)(delayed(convert_mp3_to_wav)(birdaudio, birdName, sr,) 
                                       for birdaudio in birdsaudiolist)
                
        if do_print: print(f"Conversion duration: {time.time()-start_time}s\n")
        

if __name__ == "__main__":
    conversion_function(BIRDS_DIR_LIST, SR, do_print=True)