import os
#import librosa
#import librosa.display
import numpy as np
from scipy.io import wavfile
import parselmouth.praat
from parselmouth.praat import call
import glob
cur_dir = os.getcwd()

def readGreek():
    data = []

    print("reading in the Greek data")

    for wav_file in glob.glob(cur_dir+"/data/Greek/*.wav"):
        sound = parselmouth.Sound(wav_file)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    return data

def readEnglish():
    data = []

    print("reading in the English data")

    for wav_file in glob.glob(cur_dir+"/data/English/*.wav"):
        sound = parselmouth.Sound(wav_file)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    return data



def readSpanish():
    data = []

    print("reading in the Spanish data")

    for wav_file in glob.glob(cur_dir+"/data/Spanish/es_co_female/*.wav"):
        sound = parselmouth.Sound(wav_file)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    for wav_file in glob.glob(cur_dir+"/data/Spanish/es_co_male/*.wav"):
        sound = parselmouth.Sound(wav_file)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    return data

def readPolish():
    data = []

    print("reading in the Polish data")

    for mp3 in glob.glob(cur_dir+"/data/Polish/Polish1/*.mp3"):
        sound = parselmouth.Sound(mp3)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    for mp3 in glob.glob(cur_dir+"/data/Polish/Polish2/*.mp3"):
        sound = parselmouth.Sound(mp3)
        #extract features, add them to an array
        data.append(0)
        if len(data) % 100 == 0:
            print(len(data), " files read in")
    return data

#polishdata = readPolish()
greekdata = readGreek()
englishdata = readEnglish()
spanishdata = readSpanish()
