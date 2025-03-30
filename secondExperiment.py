# Maneth P, March 2025

# libraries for general data analysis
import numpy as np
import matplotlib.pyplot as plt
import pylab

# libraries for wav file analysis
from scipy.io import wavfile
from scipy.fft import *
from scipy.fftpack import fft

# define files and colors
files = ["far", "near", "approach", "leave"]
files = ["Audio2/" + file + ".wav" for file in files]
colors = ["#2e7eff", "#29ff65", "#ff2ed5", "#ffca29"]

# get a sample object to make sure all the graphs have the same scale
n, sampleObj = wavfile.read(files[0])
globalAmpLength = len(sampleObj)
globalFreqPowerLength = len(np.abs(rfft(sampleObj)))
audioTimeScale = 5000

# function for getting data from an audio file
def getDataFromFile(audioFile):

    # access file and data type
    samplingFreq, soundObj = wavfile.read(audioFile)
    soundObjDataType = soundObj.dtype

    soundObjLength = len(soundObj)
    soundObjShape = soundObj.shape
    signalDuration =  soundObjShape[0] / samplingFreq

    # run fft (fast fourier transform) to get a frequency
    fftArray = fft(soundObj)
    numUniquePoints = np.ceil((soundObjLength + 1) / 2.0)
    fftArray = fftArray[0:int(numUniquePoints)]
    fftArray = abs(fftArray)

    fftArray = fftArray / float(soundObjLength)
    fftArray = fftArray ** 2

    if soundObjLength % 2 == 1: # odd number of points in fft
        fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2
    else: # even number of points in fft
        fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2

    # define amplitude, frequency, and power from data analysis
    amplitude = soundObj[:globalAmpLength]
    frequency = np.clip(np.abs(rfft(soundObj))[:globalFreqPowerLength], 0, None)
    dominantFreq = rfftfreq(soundObjLength, 1 / samplingFreq)[np.argmax(frequency)]
    power = 10 * np.log10(fftArray)[:globalFreqPowerLength]

    # return everything in a 2d array
    return [[amplitude, frequency, power, dominantFreq], [samplingFreq, soundObjDataType, soundObjLength, soundObjShape, signalDuration]]

# get data from each significant file
far = getDataFromFile(files[0])[0]
near = getDataFromFile(files[1])[0]
approach = getDataFromFile(files[2])[0]
leave = getDataFromFile(files[3])[0]

plotSpacing = 0.5

# create four figures, three graphs each, for twelve graphs in total
f1 = plt.figure("Near (Blue) VS Far (Green) Comparison")
f1.suptitle("Near (Blue) VS Far (Green) Comparison")

plt.subplot(3, 1, 1)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[0]))[:len(far[0])], far[0], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[0]))[:len(far[0])], near[0], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[1]))[:len(far[1])], getDataFromFile(files[0])[0][1], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[1]))[:len(far[1])], getDataFromFile(files[1])[0][1], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (kHz)")

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[2]))[:len(far[2])], far[2], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[2]))[:len(far[2])], near[2], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power (dB)")

plt.subplots_adjust(hspace = plotSpacing)

f2 = plt.figure("Approaching (Pink) VS Leaving (Yellow) Comparison")
f2.suptitle("Approaching (Pink) VS Leaving (Yellow) Comparison")

plt.subplot(3, 1, 1)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[0]))[:len(approach[0])], approach[0], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[0]))[:len(approach[0])], leave[0], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[2])[0][1], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[3])[0][1], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[2]))[:len(approach[2])], approach[2], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[2]))[:len(approach[2])], leave[2], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power")

plt.subplots_adjust(hspace = plotSpacing)

f3 = plt.figure("Near (Blue) VS Leaving (Yellow) Comparison")
f3.suptitle("Near (Blue) VS Leaving (Yellow) Comparison")

plt.subplot(3, 1, 1)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[0]))[:len(near[0])], near[0], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[0]))[:len(near[0])], leave[0], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[1]))[:len(near[1])], getDataFromFile(files[1])[0][1], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[1]))[:len(near[1])], getDataFromFile(files[3])[0][1], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[2]))[:len(near[2])], near[2], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[2]))[:len(near[2])], leave[2], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power")

plt.subplots_adjust(hspace = plotSpacing)

f4 = plt.figure("Approaching (Pink) VS Far (Green) Comparison")
f4.suptitle("Approaching (Pink) VS Far (Green) Comparison")

plt.subplot(3, 1, 1)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[0]))[:len(approach[0])], approach[0], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[0]))[:len(approach[0])], far[0], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[2])[0][1], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[0])[0][1], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[2]))[:len(approach[2])], approach[2], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[2]))[:len(approach[2])], far[2], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power")

plt.subplots_adjust(hspace = plotSpacing)

# plot everything at the end so they show up as windows
plt.show()
