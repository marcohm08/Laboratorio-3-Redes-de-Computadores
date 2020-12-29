# Laboratorio 3 Redes de Computadores
# Marco Hernandez

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy import interpolate
from scipy.io import wavfile
from scipy.fftpack import fft,fftshift,fftfreq
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import copy

import sys


def amCarrier(t, k = 1, fc = 10000):
    return k*np.cos(2*np.pi*fc*t)

# Función de muestreo
def sinc(t, compress=1):
    """
    t: vector de tiempo (evitar el cero)
    compress: factor de compresión del vector de tiempo
              > 1 comprime el tiempo, 
    """

    return np.sin(compress*np.pi*t)/(compress*np.pi*t)

def sincf(fx,t,n,sampF):
    return fx[n]*sinc((t*1-n*1/sampF),sampF)

def signalConstruct(sampleStep,Fs,duration):
    tn = np.arange(0, duration, sampleStep)# Discrete x array (or discrete time)
    fs = amCarrier(tn) # Discrete samples array
    sincs = [sincf(fs,tn,n = n,sampF=Fs) for n in range(0,len(fs))]
    sincs = np.array(sincs)
    sincs = np.sum(sincs, axis=0)
    plt.figure(6)
    sincs = np.array(sincs)
    sincs = np.sum(sincs, axis=0)
    plt.plot(tn, sincs, 'r--')
    plt.grid(True)
    plt.title("fi*sinc(t-ti)")
    plt.xlabel("time (s)")
    return sincs

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
    
    def signalExtract(self, f):
        Fs = 2*f
        sampleStep = float(1/Fs)
        tn = np.arange(0, self.duration, sampleStep)# Discrete x array (or discrete time)
        numberSamples = len(tn)
        interpolatedF = interpolate.interp1d(self.timeDomain, self.data)
        self.sampleTimeRange = np.linspace(0, self.duration, numberSamples)
        self.data = interpolatedF(self.sampleTimeRange)
        self.sourceRate = numberSamples/self.duration
    
    def amModulation(self):
        carrierData = amCarrier(self.sampleTimeRange)
        self.data = self.data * carrierData



    #def carrierDataGenerate(self,ft , f, fs):







if __name__ == "__main__":

    wave = Signal()
    wave.readWavFile("Laboratorio-3-Redes-de-Computadores/prueba.wav")
    wave.plotGraph(1,wave.timeDomain,wave.data,xLabel="tiempo(s)",yLabel="f(t)",title = 'A')

    f = 10000
    f_am = amCarrier(wave.timeDomain)
    
    fourier = np.fft.fft(f_am)
    freqDomain = fftshift(fftfreq(len(wave.timeDomain),1/len(wave.timeDomain)))

    wave.signalExtract(f)
    wave.amModulation()
    wave.fourierTransform()
    wave.plotGraph(2,wave.freqDomain,np.abs(wave.fourierData),xLabel="Frecuancias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier de señal modulada AM')
    

    plt.show()






