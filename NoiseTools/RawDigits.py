#Shamelessly copied from Tracy's sigproc_tools

# numpy is the source of all life in python
import numpy as np
import logging

# An object for handling RawDigits from art root files

class RawDigit:
    """
    RawDigit: Emulates RawDigits as read into python via "uproot". Generally this means that we are given the 
    "events" folder in the input root file which contains the RawDigits per event. We can then access each by 
    event number 
    """
    def __init__(self, EventsFolder, Producer):
        """
        args: EventsFolder is the folder containing the desired RawDigits by event
              Producer is the path to the RawDigits for uproot to decode when looking them up
        """
        self.EventsFolder = EventsFolder
        self.Producer     = Producer
        self.Obj          = EventsFolder.array(self.Producer+"obj",flatten=True)
        self.Mask         = [ False if x > 56000 else True for x in self.GetChannels(0, FullList=True) ]
        self.EmptyCount   = np.size(self.Mask) - np.count_nonzero(self.Mask)
        logging.debug('There are ' + str(self.EmptyCount) + ' masked channels.')

    def NumEvents(self):
        numEvents = len(self.Obj)
        return numEvents
        
    def NumChannels(self, EventNum):
        nChannels = self.Obj[EventNum]
        nChannels -= self.EmptyCount
        return int(nChannels)
    
    def NumTicks(self, EventNum, ChannelNum=0):
        samples = self.EventsFolder.array(self.Producer+"obj.fSamples",entrystart=EventNum,entrystop=EventNum+1,flatten=True)
        return int(samples[ChannelNum])
    
    def GetWaveforms(self, EventNum):
        """
        Plan: Provided the RawDigits exists for a given event (e.g. in Multi-TPC readout an event may have no RawDigits),
              we can look up the information to pull out the waveform from the data block. Interestingly, each waveform 
              will begin with a count (4096) and end with a guard (0) - except there is no count for the first and no guard
              for the last. So the below contortions are done to allow resizing and dropping of this extraneous info
        """
        # First check to see if this event has an entry (can happen in multiTPC readout)
        if self.NumChannels(EventNum) > 0:
            nTicks    = self.NumTicks(EventNum)
            nChannels = self.NumChannels(EventNum)
            Waveforms = self.EventsFolder.array(self.Producer+"obj.fADC",entrystart=EventNum,entrystop=EventNum+1,flatten=True)[0]
            return np.array(Waveforms)[self.Mask]
        else:
            return numpy.zeros(shape=(1,1))
        
    def GetChannels(self, EventNum, FullList=False):
        channels = self.EventsFolder.array(self.Producer+"obj.fChannel",entrystart=EventNum,entrystop=EventNum+1,flatten=True)
        if not FullList: channels = channels[self.Mask]
        return channels
