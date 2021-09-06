# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:57:32 2016

@author: ORCHISAMA
"""

from __future__ import division
import numpy as np
from scipy.io.wavfile import read
from LBG import lbg
from mel_coefficients import mfcc
from LPC import lpc
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

import librosa.display

def training(nfiltbank, orderLPC):
    nSpeaker = 4
    nCentroid = 16
    codebooks_mfcc = np.empty((nSpeaker,nfiltbank,nCentroid))
    codebooks_lpc = np.empty((nSpeaker, orderLPC, nCentroid))
    directory = os.getcwd() + '/train';
    fname = str()

    for i in range(nSpeaker):
        fname = '/s' + str(i+1) + '.wav'
        print('Now speaker ', str(i+1), 'features are being trained' )
        (fs,s) = read(directory + fname)
 
        lpc_coeff = lpc(s, fs, orderLPC)
 
        codebooks_lpc[i,:,:] = lbg(lpc_coeff, nCentroid)
        plt.figure(i)
        plt.title('Codebook for speaker ' + str(i+1) + ' with ' + str(nCentroid) +  ' centroids')
        for j in range(nCentroid):
            plt.subplot(211)
            
            plt.stem(codebooks_mfcc[i,:,j])
            plt.ylabel('MFCC')
            plt.subplot(212)
            markerline, stemlines, baseline = plt.stem(codebooks_lpc[i,:,j])
            plt.setp(markerline,'markerfacecolor','r')
            plt.setp(baseline,'color', 'k')
            plt.ylabel('LPC')
            plt.axis(ymin = -1, ymax = 1)
            plt.xlabel('Number of features')
            print(codebooks_mfcc[i,:,j])
    plt.show()
    print('Training complete')
 
    return (codebooks_mfcc, codebooks_lpc)
    
    
