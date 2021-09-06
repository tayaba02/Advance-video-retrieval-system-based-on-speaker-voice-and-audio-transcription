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
            print(thisdict[word])
    

paths =["./dataset/alice/","./dataset/Daisy/","./dataset/Jhon/","./dataset/jane/"]

for x in range(1,3):
    AUDIO_FILE = (paths[0]+str(x)+".wav")
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source)   
    try: 
        data = r.recognize_google(audio)
        print("The audio file contains: " + data)
        print(preprocess(data,x))
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio")
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech  Recognition service; {0}".format(e))

for x in range(3,5):
    AUDIO_FILE = (paths[1]+str(x)+".wav")
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source)   
    try: 
        data = r.recognize_google(audio)
        print("The audio file contains: " + data)
        print(preprocess(data,x))
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio")
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech  Recognition service; {0}".format(e))
        
for x in range(7,9):
    AUDIO_FILE = (paths[2]+str(x)+".wav")
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source)   
    try: 
        data = r.recognize_google(audio)
        print("The audio file contains: " + data)
        print(preprocess(data,x))
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio")
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech  Recognition service; {0}".format(e))
        
for x in range(5,7):
    AUDIO_FILE = (paths[3]+str(x)+".wav")
    r = sr.Recognizer() 
    with sr.AudioFile(AUDIO_FILE) as source: 
        audio = r.record(source)   
    try: 
        data = r.recognize_google(audio)
        print("The audio file contains: " + data)
        print(preprocess(data,x))
  
    except sr.UnknownValueError: 
        print("Google Speech Recognition could not understand audio")
  
    except sr.RequestError as e: 
        print("Could not request results from Google Speech  Recognition service; {0}".format(e))


words = ["good","become","chemistry"]
query(words)

