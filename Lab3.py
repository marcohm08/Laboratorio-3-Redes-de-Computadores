# Laboratorio 3 Redes de Computadores
# Marco Hernandez
# 19.318.862-1

import scipy
import scipy.io
import scipy.misc
from scipy import signal
from scipy import interpolate
from scipy import integrate
from scipy.io import wavfile
from scipy.fftpack import fft,fftshift,fftfreq,ifft
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import copy

import sys

# AM Carrier signal function
# Input:
#   t: time value or values to calculate the function
#   fc: Carrier frequency
#   k: constant, 1 by default
# Output : value or array with values of the carrier function in t
def amCarrier(t,fc, k = 1):
    return k*np.cos(2*np.pi*fc*t)

# Low pass filter Function
# Input:
#   data: function data
#   freqData: frequency domain of the data
#   threshold: max frequency allowed 
# Output: data filtered by frequency
def FirFreqFilter(data, freqData, threshold):
    filteredSignal = copy.deepcopy(data) # se copia el arreglo
    for i in range(0,len(freqData)):
        if np.abs(freqData[i]) > threshold: # Condicion de filtrado
            filteredSignal[i] = 0
    return filteredSignal


class Signal:
    def __init__(self):
        self.path = ""
        self.data = []
        self.duration = 0
        self.sourceRate = 0
        self.fourierData = []
        self.timeDomain = []
        self.freqDomain = []
    
    # Method to read a wavfile and extract the initial info
    # Input:
    #   filepath: path of where the desired wav file is allocated
    def readWavFile(self,filePath):
        self.path = filePath
        self.sourceRate, self.data = wavfile.read(filePath)
        audioArrayLenght = len(self.data)
        self.duration = audioArrayLenght / self.sourceRate
        self.timeDomain = np.linspace(0,self.duration,audioArrayLenght)
    
    # Method to plot a grapf using matplotlib
    # Input
    #   number: number of plot
    #   xData: data of the x axis
    #   yData: data oj the y axis
    #   xlabel: name of the x axis
    #   ylabel: name of the y axis
    #   titlte: title of the plot
    def plotGraph(self,number, xData, yData, xLabel = None, yLabel = None, title = None):
        plt.figure(number)
        plt.plot(xData,yData)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)

    #Method to calculate the fourier transform and the frequency domain
    def fourierTransform(self):
        self.fourierData = fft(self.data)
        dt = 1/float(self.sourceRate)
        self.freqDomain = fftfreq(len(self.data),dt)
    
    # Method to interpolate the data wich will be modulated
    # Input:
    #   f: frecuency of the carrier signal
    def signalExtract(self, f):
        # This is the frecuency wich is used to extract samples of the function to rebuild the signal, according to the sampling theorme it must be at leas 2 times the frequency, in this time it is 2.6
        Fs = 2.6*f 
        sampleStep = float(1/Fs)
        tn = np.arange(0, self.duration, sampleStep)
        numberSamples = len(tn)
        # This is to recreate the function that represents the wav file data
        interpolatedF = interpolate.interp1d(self.timeDomain, self.data)
        self.sampleTimeRange = np.linspace(0, self.duration, numberSamples)
        self.data = interpolatedF(self.sampleTimeRange)
        self.sourceRate = numberSamples/self.duration
    
    # Method to apply the AM modulation to the signal
    # Input: 
    #   freq: frequency of the carrier signal
    def amModulation(self,freq):
        carrierData = amCarrier(self.sampleTimeRange,freq)
        self.data = self.data * carrierData
        self.timeDomain = np.linspace(0, self.duration, len(self.data))

    # Method to apply the FM modulation to the signal
    # Input: 
    #   freq: frequency of the carrier signal
    def fmModulation(self,freq, k = 1):
        iSum = integrate.cumtrapz(self.data,self.sampleTimeRange,initial=0)
        self.data = np.cos(2*np.pi*freq*self.sampleTimeRange+ k*iSum)
        self.timeDomain = np.linspace(0, self.duration, len(self.data))

    # Method to apply the AM signal demodulation
    # Inputs:
    #   f: frequency of the carrier signal
    #   filterfreq: frecuency to the low pass filter
    #   k: constant, 1 by default
    def amDemodulation(self, f, filterfreq, k = 1):
        self.data = k*self.data*np.cos(2*np.pi*f*self.sampleTimeRange)
        self.fourierTransform()
        self.fourierData = FirFreqFilter(self.fourierData,self.freqDomain,filterfreq)
        self.data = ifft(self.fourierData)
        
    
    # Method to write a new wav file with the current data of the signal object
    # Input:
    #   name: name of the output file
    def write(self, name=None):
        data = np.asarray(np.real(self.data), dtype=np.int16)
        wavfile.write(name, int(self.sourceRate), data)


if __name__ == "__main__":

    waveAM = Signal()
    waveAM.readWavFile("Laboratorio-3-Redes-de-Computadores/prueba.wav")
    waveAM.plotGraph(1,waveAM.timeDomain,waveAM.data,xLabel="tiempo(s)",yLabel="f(t)",title = 'Grafico de la señal en funcion del tiempo')
    waveAM.fourierTransform()
    waveAM.plotGraph(2,waveAM.freqDomain,np.abs(waveAM.fourierData),xLabel="Frecuancias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier de señal modulada AM')

    f = 20000

    waveAM.signalExtract(f)
    AmCarrierData = amCarrier(waveAM.sampleTimeRange,f)
    carrierTransform = fft(AmCarrierData)
    waveAM.amModulation(f)
    waveAM.plotGraph(3,waveAM.timeDomain,waveAM.data,xLabel="tiempo(s)",yLabel="f(t)",title = 'Grafico de la señal modulada AM en funcion del tiempo')
    waveAM.fourierTransform()
    waveAM.plotGraph(4,waveAM.freqDomain,np.abs(waveAM.fourierData),xLabel="Frecuencias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier de señal modulada AM')
    waveAM.plotGraph(5,waveAM.timeDomain,AmCarrierData,xLabel="tiempo(s)",yLabel="f(t)",title = 'Grafico de la señal portadora en funcion del tiempo')
    waveAM.plotGraph(6,waveAM.freqDomain,np.abs(carrierTransform),xLabel="Frecuencias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier señal portadora')

    waveAM.amDemodulation(f, 4000)
    waveAM.plotGraph(7,waveAM.freqDomain,np.abs(waveAM.fourierData),xLabel="Frecuencias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier de señal demodulada AM')
    plt.xlim([-4000,4000])

    waveAM.plotGraph(8,waveAM.timeDomain,waveAM.data.real,xLabel="tiempo(s)",yLabel="f(t)",title = 'Grafico de la señal en funcion del tiempo demodulada')

    waveAM.write("Señal_AM.wav")

    fm = 10000

    waveFM = Signal()
    waveFM.readWavFile("Laboratorio-3-Redes-de-Computadores/prueba.wav")
    waveFM.signalExtract(fm)
    waveFM.fmModulation(fm)
    waveAM.plotGraph(9,waveFM.timeDomain,np.abs(waveFM.data),xLabel="Frecuencias(Hz)",yLabel="F(w)",title = 'Grafico de la señal modulada FM en funcion del tiempo')
    waveFM.fourierTransform()
    waveFM.plotGraph(10,waveFM.freqDomain,np.abs(waveFM.fourierData),xLabel="Frecuencias(Hz)",yLabel="F(w)",title = 'Transformada de Fourier de señal modulada FM')

    plt.show()
