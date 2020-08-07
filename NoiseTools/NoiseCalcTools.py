import numpy as np
import scipy.signal as signal
import pandas as pd
import logging
from RawDigits import RawDigit

def RMSCalc(RawDigits, NumEvents=50):
    # This function calculates the RMS for each channel and returns an average over the
    # number of events. The RawDigits argument is an object which serves as an interface 
    # to retrieving the raw digits from the input ROOT file.
    
    nChannels = RawDigits.NumChannels(0)               # The number of channels per event.
    nTicks = RawDigits.NumTicks(0)                     # The number of ticks per waveform.
    nEvents = RawDigits.NumEvents()                    # The number of events in the file.
    RawWaveforms  = np.zeros((nChannels,nTicks))       # Empty array for raw digits.

    # Unfortunately it is not possible to do this all at once since there are up to
    # 55,000 channels and each waveform is 4096 ticks long. This means we need to
    # perform our calculation in stages. First we determine how many events to include
    # in the calculation: either the argument of the function or the number in the file,
    # whichever is smallest. For each event we load up the waveforms, find the pedestal
    # as the median of the waveform, then subtract the pedestal. The resulting quantity
    # is squared and added to RawWaveforms. Finally we divide by N to get the 'mean-square'
    # and take the square root to get the final RMS.
    N = NumEvents if NumEvents < nEvents else nEvents
    for n in range(N):
        if n % 10 == 0: print('Processing (RMS) event ' + str(n) + '...')
        # Each quantity is (nChannels,nTicks)
        Waveforms = RawDigits.GetWaveforms(n)
        Pedestals = np.median(Waveforms, axis=-1)
        WaveLessPeds = Waveforms - Pedestals.reshape((Pedestals.shape)+(1,))
        RawWaveforms[:,:] += np.square(WaveLessPeds)
    RawWaveforms /= N 
    RMS = np.sqrt(np.mean(RawWaveforms,axis=-1))
  
    # Now we return RMS, which is a 1D numpy array of length nChannels containing the
    # RMS value for each channel.
    return RMS

def PowerCalc(RawDigits, IsRaw, NumEvents=50):
    # This function calculates the power spectrum of each channel as an average over the
    # number of events. Again we use a RawDigit object as an interface to the raw digits.
    
    nChannels = RawDigits.NumChannels(0)               # The number of channels per event.
    nTicks = RawDigits.NumTicks(0)                     # The number of ticks per waveform.
    nEvents = RawDigits.NumEvents()                    # The number of events in the file.
    Spectrum  = np.zeros((nChannels,2049))             # Empty array for the spectrum.
    tmpSpectrum = np.zeros((nChannels,2049))           # Temporary array for the spectrum.

    # We first determine the number of events to use for this averaging: either the
    # number of events from the argument or the number in the ROOT file, whichever is
    # smallest. Then we begin our calculation by looping over each event. For each event
    # we retrieve the waveform and, if it's a raw waveform (not coherent noise subtracted)
    # we subtract the pedestal from the waveform. This would be redundant for the coherent
    # subtracted waveforms. Then we use the Scipy signal package to calculate the FFT for
    # each waveform and add it to Spectrum. Finally after looping over each event we divide
    # Spectrum by the number of events to get the power spectrum averaged over the events
    # for each channel.
    N = NumEvents if NumEvents < nEvents else nEvents
    for n in range(N):
        if n % 10 == 0: print('Processing (power) event ' + str(n) + '...')
        # Each quantity below is (nChannels,nTicks).
        Waveforms = RawDigits.GetWaveforms(n)
        if IsRaw:
            Pedestals = np.median(Waveforms, axis=-1)
            WaveLessPeds = Waveforms - Pedestals.reshape((Pedestals.shape)+(1,))
            Frequency, tmpSpectrum = signal.periodogram(WaveLessPeds, 1/0.4, axis=1)
        else: Frequency, tmpSpectrum = signal.periodogram(WaveLessPeds, 1/0.4, axis=1)
        Spectrum += tmpSpectrum
    Spectrum /= N 

    # Finally we return the 1D numpy array of the frequencies and the 2D numpy array
    # containing the power spectrum for each channel (nChannels,2049).
    return Frequency, Spectrum

def MeanPower(Power, Map, MiniCrate):
    # This function calculates the mean power spectrum for the requested mini-crate. We
    # require the power spectrum of each channel, the channel map, and the name of the
    # mini-crate (e.g. 'WE05'). A boolean mask is used to select the channels which
    # are in the mini-crate, and then we average the power spectrum over the selected
    # channels.
    
    Mask = Map['fCrate'] == MiniCrate
    Spectrum = np.mean(Power[Mask], axis=0)
    
    # Spectrum is a 1D numpy array containing the mean power spectrum for the requested
    # mini-crate.
    return Spectrum

def PeakFind(Frequency, Power, fLow=1, fHigh=800):
    # This function calculates the frequency (kHz) and height of the peak in the power
    # spectrum in the range of fLow to FHigh kHz. 
    
    # First we create a boolean mask to hone in on the frequency region of interest.
    # We then apply the range mask and locate the max height of the spectrum in the
    # region of interest.
    RangeMask = [ True if x < fHigh and x > fLow else False for x in 1000*Frequency]
    Power_Selected = Power[RangeMask]
    ArgMax = np.argmax(Power_Selected)
    PeakFrequency = 1000*Frequency[RangeMask][ArgMax]
    PeakPower = Power_Selected[ArgMax]
    GlobalArgMax = np.argwhere(Power == PeakPower)[0,0]
    
    # The frequency at which the power spectrum is maximum as well as the peak power
    # are returned.
    return PeakFrequency, PeakPower, GlobalArgMax
