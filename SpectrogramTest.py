import numpy as np
import matplotlib.pyplot as plt
import pylab
from scipy.io import wavfile
from scipy.fft import *
from scipy.fftpack import fft

testFile = "Audio/orig.wav"

colors = ["#2e7eff", "#29ff65", "#ff2ed5", "#ff2e1f"]

samplingFreq, soundObj = wavfile.read(testFile)
soundObjDataType = soundObj.dtype

soundObjLength = len(soundObj)
soundObjShape = soundObj.shape
signalDuration =  soundObj.shape[0] / samplingFreq
soundObjMono = soundObj[:,0]

def getTimeScale(data):
    print(float(data.shape[0]))
    print(samplingFreq)
    timeArray = np.arange(0, float(data.shape[0]), 1)
    timeArray = (timeArray / samplingFreq) * 1000
    return timeArray

fftArray = fft(soundObjMono)
numUniquePoints = np.ceil((soundObjLength + 1) / 2.0)
fftArray = fftArray[0:int(numUniquePoints)]
fftArray = abs(fftArray)

fftArray = fftArray / float(soundObjLength)
fftArray = fftArray ** 2

if soundObjLength % 2 == 1: # odd number of points in fft
    fftArray[1:len(fftArray)] = fftArray[1:len(fftArray)] * 2
else: # even number of points in fft
    fftArray[1:len(fftArray) -1] = fftArray[1:len(fftArray) -1] * 2

#freqArray = numpy.arange(0, numUniquePoints, 1.0) * (samplingFreq / soundObjLength)

xf = rfftfreq(soundObjLength, 1 / samplingFreq)

#frequency = freqArray / 1000
amplitude = soundObjMono
frequency = np.abs(rfft(soundObjMono))
dominantFreq = xf[np.argmax(frequency)]
power = 10 * np.log10(fftArray)

np.savetxt("timeTest1.txt", frequency)

plt.figure(10)
plt.plot(getTimeScale(frequency), frequency, color=colors[0])
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (kHz)")

plt.figure(20)
plt.plot(xf, frequency, color=colors[2])
plt.xlabel("Time (ms)")
plt.ylabel("Frequency (kHz)")

print("owiefjoi3jfwe")
print(getTimeScale(frequency))
print(getTimeScale(amplitude))


plt.show()

"""
freqArrayLength = len(freqArray)
print(freqArrayLength)
np.savetxt("freqData.txt", freqArray, fmt='%6.2f')
print(len(fftArray))
np.savetxt("fftData.txt", fftArray)"
"""