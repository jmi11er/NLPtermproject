#!/usr/bin/env python
# coding: utf-8

# # **Battle of the Languages**
# here is our code, blah blah blah, we can write more stuff here later
# please feel free to play with the groupings and cells, I just did some initial stuff I thought  might be helpful, but it might need to be broken up more

# Importing the Data from the various databases:
# 
# https://github.com/festvox/datasets-CMU_Wilderness One of the ones Emily recommended, has like 700 languages, seems like it was mined from people reading the new testament. has polish, spanish, english, (probably has greek but it's labeled by the language in that language so I was not positive what I was looking for)
# 
# https://openslr.org/resources.php The other one Emily recommended instead of the UPenn one, had a brief look and seems like it might be more helpful for spanish/english.

# In[ ]:


# YOUR FILEPATH HERE
FILEPATH = "/Users/eviprousanidou/Desktop/BC/Natural Language Programming/final/" 


# In[ ]:


import glob
import re
import parselmouth
from parselmouth.praat import call
import numpy as np

from os.path import join


import parselmouth, matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier


# <h1> Get Features

# In[ ]:


def getFeatures(wav_file):
    # get duration, mean pitch, mean intensity
    sound = parselmouth.Sound(wav_file)
    # pitch
    pitch = call(sound, "To Pitch", 0, 75, 600) 
    
    # new ==============
    pitch_stdev = call(pitch, "Get standard deviation", 0, 0, "Hertz")
    
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")
    min_pitch = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")
    max_pitch = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
    
    # intensity
    intensity = call(sound, "To Intensity", 75, 0, "yes")

    # new ==============
    intensity_stdev = call(intensity, "Get standard deviation", 0, 0)
    
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")
    min_intensity = call(intensity, "Get minimum", 0, 0, "Parabolic")
    max_intensity = call(intensity, "Get maximum", 0, 0, "Parabolic")
    

    duration = call(sound, "Get total duration")

    # get mean features for vowels and consonants
    formant = call(sound, "To Formant (burg)", 0, 5, 5500, 0.025, 50)
    tg_file = re.sub("wav", "TextGrid", wav_file)
    textgrid = call("Read from file", tg_file)
    intv = call(textgrid, "Get number of intervals", 1)
    
    # new ==============
    # get mfccs
    mfccs = call(sound, "To MFCC", 12, 0.015, 0.005, 100, 100, 0).to_array()
    mfccs_avg = [np.mean(mfccs[i]) for i in range(13)]
    
    
#     vowels = 0
#     consonants = 0
#     f1_vowels = 0
#     dur_vowels = 0
    


#     for i in range(1, intv):
#         phone = call(textgrid, "Get label of interval", 1, i)
#         # vowels
#         if phone == 'sil': continue
#         if re.match('[AEIOU]', phone):
#             vowels += 1
#             vowel_onset = call(textgrid, "Get starting point", 1, i)
#             vowel_offset = call(textgrid, "Get end point", 1, i)
#             midpoint = vowel_onset + ((vowel_offset - vowel_onset) / 2)
#             f1_vowels += call(formant, "Get value at time", 1, midpoint, "Hertz", "Linear")
#             dur_vowels += vowel_offset - vowel_onset
    
#     f1_vowels = f1_vowels / vowels if vowels > 0 else 0
#     dur_vowels = dur_vowels / vowels if vowels > 0 else 0
    
    

    results = [
                pitch_stdev,
                mean_pitch, 
                min_pitch,
                max_pitch,
                intensity_stdev,
                mean_intensity, 
                min_intensity,
                max_intensity,
                duration,
                mfcc_avg,
#                 f1_vowels, 
#                 dur_vowels,
            ]
   
    return results


# <h1> Read Greek Data
#     

# In[ ]:


counter = 0 

greek=[]

for wav_file in glob.glob(join(FILEPATH, "data/Greek/*.wav")):
    
    # print progress
    counter += 1
    if counter % 100 == 0:
        print(counter, wav_file)

    results = getFeatures(wav_file)
    greek.append(results)


# <h1> Read Polish / Czech Data

# In[ ]:


counter = 0 

czech=[]

for wav_file in glob.glob(join(FILEPATH, "data/Czech/*.wav")):
    
    # print progress
    counter += 1
    if counter % 100 == 0:
        print(counter, wav_file)

    results = getFeatures(wav_file)
    czech.append(results)


# <h1> Read Spanish Data

# In[ ]:


counter = 0 

spanish=[]

for wav_file in glob.glob(join(FILEPATH, "data/Spanish/*.wav")):
    
    # print progress
    counter += 1
    if counter % 100 == 0:
        print(counter, wav_file)

    results = getFeatures(wav_file)
    spanish.append(results)


# <h1> Read English Data

# In[ ]:


counter = 0 

english=[]

for wav_file in glob.glob(join(FILEPATH, "data/English/*.wav")):
    
    # print progress
    counter += 1
    if counter % 100 == 0:
        print(counter, wav_file)

    results = getFeatures(wav_file)
    english.append(results)


# <h1> Feature Selection
#     

# <p> Comment out any features we don't want to include

# In[ ]:


selection = [
    0, # pitch_stdev
    1, # mean_pitch
    2, # min_pitch
    3, # max_pitch
    4, # intensity_stdev
    5, # mean_intensity
    6, # min_intensity
    7, # max_intensity
    8, # duration
    9, # mfcc_avg
]

npdata = npdata[:, selection]
npdata.shape


# <h1> Scoring Metrics
#     

# In[ ]:


scoring_metrics = ['accuracy', 'precision', 'recall', 'f1']


# <h1> Models

# In[ ]:


models = [
            GaussianNB(), 
            MLPClassifier(max_iter=300),
            LinearSVC(), 
            # Feedforward Neural Network
            QuadraticDiscriminantAnalysis(),
            LogisticRegression(),
            RandomForestClassifier(max_depth=4),  # try more max_depth
            DecisionTreeClassifier(), 
            KNeighborsClassifier(n_neighbors=3), # try more number of neighbors
            AdaBoostClassifier(n_estimators=100)
        ]


# <h1> Test each Model
#     

# In[ ]:


for model in models:

    # print model name
    model_name = str(type(model))
    model_name = model_name[model_name.rfind('.')+1:-2]
    print('\n' + model_name + '\n')

    # train and cross validate with 5 folds
    scores = cross_validate(model, npdata, nptarget, cv=5, scoring=scoring_metrics)
    for score_name, score_value in scores.items():
        if "test" in score_name:
            print(score_name, "\t", np.round(np.mean(score_value), 4))


# Compiling said data to compatible formats and whatnot

# In[2]:


#insert code here


# Feature Extraction and beyond...
