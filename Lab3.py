# Laboratorio 3 Redes de Computadores
# Marco Hernandez

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft2,fftshift
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import copy

import sys


class Signal:
    def __init__(self):
        self.path = ""
        self.data = []
        self.duration = 0
        self.sourceRate = 0
        self.fourierData = []
        self.timeDomain = []
        self.freqDomain = []
    
    def readWavFile(self,filePath):
        self.path = path
        self.sourceRate, self.data = wavfile.read(filePath)
        audioArrayLenght = len(self.data)
        self.duration = audioArrayLenght / self.sourceRate
        self.timeDomain = np.linspace(0,self.duration,audioArrayLenght)








