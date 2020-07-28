# Python includes
import uproot
import sys
import numpy as np
import scipy as sp
import pandas as pd
import logging
import signal as sg
from os import path

# Custom includes
sys.path.insert(0, './NoiseTools/')
from RawDigits import RawDigit
from NoiseCalcTools import RMSCalc, PowerCalc, PeakFind
from NoiseHelperTools import SigintHandler, ReturnConfig, Decode
from NoisePlottingTools import PlotRMS, PlotPower, PlotPowerAsHeatmap
from DatabaseTools import BuildMapDataFrame

def AnalyzeHalf(Events, cfg, Side='East'):
    #Gather data and channel map
    RawDigits_Raw = RawDigit(Events, cfg['Path']['DAQName_Raw'])
    RawDigits_Uncor = RawDigit(Events, cfg['Path']['DAQName_Uncor'])
    ChannelList = RawDigits_Raw.GetChannels(0)
    u, c = np.unique(ChannelList, return_counts=True)
    logging.debug('Duplicate channels: ' + str(u[c > 1]))
    logging.debug('Duplicate channel counts: ' + str(c[c > 1]))
    logging.debug('ChannelList length: ' + str(len(ChannelList)))

    MapExists = path.exists(cfg['Data'][Side+'MapName']+'.csv')
    if not MapExists: BuildMapDataFrame(ChannelList, Name=cfg['Data']['MapName'])
    Dataframe = pd.read_csv(cfg['Data'][Side+'MapName']+'.csv')
    logging.debug('Dataframe shape: ' + str(Dataframe.shape))
    logging.debug('Dataframe head: ' + str(Dataframe.head))

    #Temp Debugging
    MiniCrateList = Dataframe.fCrate.unique()
    for MiniCrate in MiniCrateList:
        Selection = Dataframe.loc[ Dataframe['fCrate'] == MiniCrate ]
        logging.debug(MiniCrate + ': ' + str(Selection.shape))
        u2, c2 = np.unique(Selection.fChannel.to_numpy(), return_counts=True)
        logging.debug('Duplicate channels: ' + str(u2[c2 > 1]))
        logging.debug('Duplicate channel counts: ' + str(c2[c2 > 1]))
        #logging.debug(Selection.to_string())

    #Calculate RMS for each channel
    RMSRaw = RMSCalc(RawDigits_Raw, NumEvents=cfg['Analysis']['Events'])
    RMSUncor = RMSCalc(RawDigits_Uncor, NumEvents=cfg['Analysis']['Events'])
    RMSData = {'fID': ChannelList, 'fRMS': RMSRaw, 'fUnRMS': RMSUncor}
    tmp = pd.DataFrame(RMSData)
    Dataframe = Dataframe.merge(tmp, on='fID', how='left')
    print(Dataframe.head())
    
    #Plot RMS
    MiniCrateList = Dataframe.fCrate.unique()
    for MiniCrate in MiniCrateList:
        Selection = Dataframe.loc[ Dataframe['fCrate'] == MiniCrate ]
        PlotRMS(Selection, MiniCrate, './test/')

    #Calculate the power spectrum for each channel
    Frequency, PowerRaw = PowerCalc(RawDigits_Raw, True, NumEvents=cfg['Analysis']['Events'])
    Frequency, PowerUncor = PowerCalc(RawDigits_Uncor, True, NumEvents=cfg['Analysis']['Events'])

    #Plot power spectrums for each mini-crate
    for MiniCrate in MiniCrateList:
        PlotPower(Frequency, PowerRaw, PowerUncor, Dataframe, MiniCrate, './test/')

    #Locate peak frequency and associated power for each mini-crate
    PowerDict = {'fCrate': [], 'fFreq': [], 'fPow': []}
    for MiniCrate in MiniCrateList:
        fFreq, fPow = PeakFind(Frequency, PowerRaw, PowerUncor, Dataframe, MiniCrate)
        PowerDict['fCrate'].append(MiniCrate)
        PowerDict['fFreq'].append(fFreq)
        PowerDict['fPow'].append(fPow)
    PowerFrame = pd.DataFrame(PowerDict)
    return PowerFrame

def main():
    # Preliminary configuration
    sg.signal(sg.SIGINT, SigintHandler)
    cfg = ReturnConfig('Config.yaml')
    logging.basicConfig(filename=cfg['Miscellaneous']['LogPath'] + cfg['Miscellaneous']['LogName'], level=logging.DEBUG, filemode='w')
    logging.warning('Logging service has started.')

    # Connect to ROOT file
    EastFileToProcess = Decode(str(cfg['Data']['EastRun']))
    WestFileToProcess = Decode(str(cfg['Data']['WestRun']))
    EastData = uproot.open(EastFileToProcess)
    EastEvents = EastData[cfg['Path']['RecoFolder']]
    WestData = uproot.open(WestFileToProcess)
    WestEvents = WestData[cfg['Path']['RecoFolder']]

    # Run the analysis on the single cryostat
    EastPower = AnalyzeHalf(EastEvents, cfg, 'East')
    WestPower = AnalyzeHalf(WestEvents, cfg, 'West')
    PowerFrame = EastPower.append(WestPower)
    print(PowerFrame)
    PowerFrame.to_csv('Power.csv', index=False)

    # Plot the power in a geographically relevant heatmap.
    PlotPowerAsHeatmap(PowerFrame,
                       cfg['SVGHeatmap']['Gradient'],
                       cfg['SVGHeatmap']['SVGBase'],
                       ZMin=cfg['SVGHeatmap']['ZMin'],
                       ZMax=cfg['SVGHeatmap']['ZMin'],
                       EmptyColor=cfg['SVGHeatmap']['EmptyColor'])

if __name__ == "__main__":
    main()
