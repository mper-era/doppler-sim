import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fft import *
from scipy.fftpack import fft

files = ["orig", "noise1", "noise2", "far", "near", "approach", "leave"]
files = ["Audio/" + file + ".wav" for file in files]
colors = ["#2e7eff", "#29ff65", "#ff2ed5", "#ffca29"]

n, sampleObj = wavfile.read(files[6])
globalAmpLength = len(sampleObj)
globalFreqPowerLength = len(np.abs(rfft(sampleObj)))
audioTimeScale = 10000

def getDataFromFile(audioFile):

    samplingFreq, soundObj = wavfile.read(audioFile)
    soundObjDataType = soundObj.dtype

    soundObjLength = len(soundObj)
    soundObjShape = soundObj.shape
    signalDuration =  soundObjShape[0] / samplingFreq

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

    amplitude = soundObj[:globalAmpLength]
    frequency = np.clip(np.abs(rfft(soundObj))[:globalFreqPowerLength], 0, None)
    dominantFreq = rfftfreq(soundObjLength, 1 / samplingFreq)[np.argmax(frequency)]
    power = 10 * np.log10(fftArray)[:globalFreqPowerLength]

    return [[amplitude, frequency, power, dominantFreq], [samplingFreq, soundObjDataType, soundObjLength, soundObjShape, signalDuration]]

noise = [(n1 + n2) / 2 for n1, n2 in zip(getDataFromFile(files[1])[0], getDataFromFile(files[2])[0])]

def subtractNoise(dataset):
    return [[n1 - n2 for n1, n2 in np.column_stack((dataset[n], noise[n]))] for n in range(len(dataset))]

far = subtractNoise(getDataFromFile(files[3])[0])
near = subtractNoise(getDataFromFile(files[4])[0])
approach = subtractNoise(getDataFromFile(files[5])[0])
leave = subtractNoise(getDataFromFile(files[6])[0])

plotSpacing = 0.5

f1 = plt.figure("Near (Blue) VS Far (Green) Comparison")
f1.suptitle("Near (Blue) VS Far (Green) Comparison")

plt.subplot(3, 1, 1)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[0]))[:len(far[0])], far[0], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[0]))[:len(far[0])], near[0], color=colors[1], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.subplot(3, 1, 2)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[1]))[:len(far[1])], getDataFromFile(files[3])[0][1], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(far[1]))[:len(far[1])], getDataFromFile(files[4])[0][1], color=colors[1], alpha=0.5)
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
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[5])[0][1], color=colors[2], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(approach[1]))[:len(approach[1])], getDataFromFile(files[6])[0][1], color=colors[3], alpha=0.5)
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
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[1]))[:len(near[1])], getDataFromFile(files[4])[0][1], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[1]))[:len(near[1])], getDataFromFile(files[6])[0][1], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Frequency")

plt.subplot(3, 1, 3)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[2]))[:len(near[2])], near[2], color=colors[0], alpha=0.5)
plt.plot(np.arange(0, audioTimeScale, audioTimeScale/len(near[2]))[:len(near[2])], leave[2], color=colors[3], alpha=0.5)
plt.xlabel("Time (ms)")
plt.ylabel("Power")

plt.subplots_adjust(hspace = plotSpacing)

plt.show()
