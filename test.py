# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:21:03 2016

@author: ORCHISAMA
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import EUDistance
from mel_coefficients import mfcc
from LPC import lpc
from train import training
import os
import speech_recognition as sr 
import re
from nltk import *
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from collections import *
import string
from string import digits
import numpy as np
import math
from textblob import TextBlob
#global variables

import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

thisdict ={}

def preprocess(data,video):
    #This for word correction. Because in speech recognition few of the world can be spelled incorrectly so we can use this.
    
    data = TextBlob(data)
    data = str(data.correct())
    data= data.rstrip("\n\r")
    p = str.maketrans("", "", string.punctuation)
    data = data.translate(p)
    d = str.maketrans('', '', digits)
    data = data.translate(d)
    data = data.split()
    stopwordremoved = []
    for x in data:
        if not x in stopwords.words('english'):
            stopwordremoved.append(x)
    for x in stopwordremoved:
        if x in thisdict:
            thisdict[x] = str(thisdict[x])+(str(video)+",")
        else:
            thisdict[x] = str(video)+","
    return thisdict


def query(words):
    for word in words:
        if word in thisdict:
            print(word +"-" + thisdict[word])
nSpeaker = 1
nfiltbank = 12
orderLPC = 15
(codebooks_mfcc, codebooks_lpc) = training(nfiltbank, orderLPC)
directory = os.getcwd() + '/test';
fname = str()
nCorrect_MFCC = 0
nCorrect_LPC = 0


def minDistance(features, codebooks):
    speaker = 0
    distmin = np.inf
    for k in range(np.shape(codebooks)[0]):
        D = EUDistance(features, codebooks[k,:,:])
        dist = np.sum(np.min(D, axis = 1))/(np.shape(D)[0]) 
        if dist < distmin:
            distmin = dist
            speaker = k
            
    return speaker
    

for i in range(nSpeaker):
    fname = '/s' + str(i+1) + '.wav'
    print('Now speaker ', str(i+1), 'features are being tested')
    (fs,s) = read(directory + fname)
    print(fs)
    #mel_coefs = mfcc(s,fs,nfiltbank)
    lpc_coefs = lpc(s, fs, orderLPC)
    #sp_mfcc = minDistance(mel_coefs, codebooks_mfcc)
    sp_lpc = minDistance(lpc_coefs, codebooks_lpc)
    
 
    print('Speaker ', (i+1), ' in test matches with speaker ', (sp_lpc+1), ' in train for training with LPC')
    fname = '/s' + str(i+1) + '.wav'
    AUDIO_FILE = ("./files"+fname)
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source)   
    try: 
        data = r.recognize_google(audio)
        print("The audio file contains: " + data)
        print(preprocess(data,i))
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio")
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech  Recognition service; {0}".format(e))
    
    
    if i == sp_lpc:
        nCorrect_LPC += 1
    



words = ["machine","computer"]
query(words)
percentageCorrect_LPC = (nCorrect_LPC/nSpeaker)*100
#print('Accuracy of result for training with LPC is ', percentageCorrect_LPC, '%')


    
