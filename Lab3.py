# Laboratorio 3 Redes de Computadores
# Marco Hernandez

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft,fftshift,fftfreq
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
        self.path = filePath
        self.sourceRate, self.data = wavfile.read(filePath)
        audioArrayLenght = len(self.data)
        self.duration = audioArrayLenght / self.sourceRate
        self.timeDomain = np.linspace(0,self.duration,audioArrayLenght)
    
    def plotGraph(self,number, xData, yData, xLabel = None, yLabel = None, title = None):
        plt.figure(number)
        plt.plot(xData,yData)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
    def fourierTransform(self):
        self.fourierData = fft(self.data)
        dt = 1/float(self.sourceRate)
        self.freqDomain = fftfreq(len(self.data),dt)



if __name__ == "__main__":
    wave = Signal()
    wave.readWavFile("prueba.wav")
    wave.plotGraph(1,wave.timeDomain,wave.data,xLabel="tiempo(s)",yLabel="f(t)",title = 'A')
    

    plt.show()






