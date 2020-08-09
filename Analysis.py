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
from NoiseCalcTools import RMSCalc, PowerCalc, MeanPower, PeakFind
from NoiseHelperTools import SigintHandler, ReturnConfig, Decode
from NoisePlottingTools import PlotRMS, PlotPower, PlotPowerAsHeatmap, PlotWithBackgroundSeparation
from SpectraTools import BackgroundSNIPCalc
from DatabaseTools import BuildMapDataFrame

def Analyze(Events, cfg, Run):
    #Gather data and channel map
    RawDigits_Raw = RawDigit(Events, cfg['Path']['DAQName_Raw'])
    RawDigits_Uncor = RawDigit(Events, cfg['Path']['DAQName_Uncor'])
    ChannelList = RawDigits_Raw.GetChannels(0)
    u, c = np.unique(ChannelList, return_counts=True)
    logging.debug('Duplicate channels: ' + str(u[c > 1]))
    logging.debug('Duplicate channel counts: ' + str(c[c > 1]))
    logging.debug('ChannelList length: ' + str(len(ChannelList)))

    MapExists = path.exists(cfg['Data']['Runs'][Run]+'.csv')
    if not MapExists: BuildMapDataFrame(ChannelList, Name=cfg['Data']['Runs'][Run])
    Dataframe = pd.read_csv(cfg['Data']['Runs'][Run]+'.csv')
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

    #Calculate RMS for each channel
    RMSRaw = RMSCalc(RawDigits_Raw, NumEvents=cfg['Analysis']['Events'])
    RMSUncor = RMSCalc(RawDigits_Uncor, NumEvents=cfg['Analysis']['Events'])
    RMSData = {'fID': ChannelList, 'fRMS': RMSRaw, 'fUnRMS': RMSUncor}
    tmp = pd.DataFrame(RMSData)
    Dataframe = Dataframe.merge(tmp, on='fID', how='left')
    
    #Plot RMS
    MiniCrateList = Dataframe.fCrate.unique()
    for MiniCrate in MiniCrateList:
        Selection = Dataframe.loc[ Dataframe['fCrate'] == MiniCrate ]
        PlotRMS(Selection, MiniCrate, cfg['Path']['Images'])

    #Calculate the power spectrum for each channel
    Frequency, PowerRaw = PowerCalc(RawDigits_Raw, True, NumEvents=cfg['Analysis']['Events'])
    Frequency, PowerUncor = PowerCalc(RawDigits_Uncor, True, NumEvents=cfg['Analysis']['Events'])

    #Plot power spectrums for each mini-crate
    for MiniCrate in MiniCrateList:
        PlotPower(Frequency, PowerRaw, PowerUncor, Dataframe, MiniCrate, cfg['Path']['Images'])

    #Locate peak frequency and associated power for each mini-crate
    PowerDict = {'fCrate': [], 'fFreq': [], 'fPow': [], 'fPowSep': [], 'fPowBack': [], 'fRatio': []}
    for MiniCrate in MiniCrateList:
        PowerRaw_Selected = MeanPower(PowerRaw, Dataframe, MiniCrate)
        PowerUncor_Selected = MeanPower(PowerUncor, Dataframe, MiniCrate)
        fFreq, fPow, ArgMax = PeakFind(Frequency, PowerRaw_Selected, cfg['Analysis']['fLow'], cfg['Analysis']['fHigh'])
        Background = BackgroundSNIPCalc(PowerRaw_Selected, nIterations=20, ApplyLLS=True)
        PlotWithBackgroundSeparation(Frequency, PowerRaw_Selected, Background, MiniCrate, Path=cfg['Path']['Images'], Suffix='')
        fPowSep = (PowerRaw_Selected - Background)[ArgMax]
        fPowBG = Background[ArgMax]
        fRatio = fPowSep / fPowBG
        PowerDict['fCrate'].append(MiniCrate)
        PowerDict['fFreq'].append(fFreq)
        PowerDict['fPow'].append(fPow)
        PowerDict['fPowSep'].append(fPowSep)
        PowerDict['fPowBack'].append(fPowBG)
        PowerDict['fRatio'].append(fRatio)
    PowerFrame = pd.DataFrame(PowerDict)
    
    # Locate the peak frequency and associated power for each channel
    ChannelPowerDict = {'fID': ChannelList, 'fFreq': [], 'fPow': [], 'fPowSep': [], 'fPowBack': [], 'fRatio': []}
    for n, PowerSpectrum in enumerate(PowerRaw):
        fFreq, fPow, ArgMax = PeakFind(Frequency, PowerSpectrum, cfg['Analysis']['fLow'], cfg['Analysis']['fHigh'])
        Background = BackgroundSNIPCalc(PowerSpectrum, nIterations=20, ApplyLLS=True)
        fPowSep = (PowerSpectrum - Background)[ArgMax]
        fPowBG = Background[ArgMax]
        fRatio = fPowSep / fPowBG
        ChannelPowerDict['fFreq'].append(fFreq)
        ChannelPowerDict['fPow'].append(fPow)
        ChannelPowerDict['fPowSep'].append(fPowSep)
        ChannelPowerDict['fPowBack'].append(fPowBG)
        ChannelPowerDict['fRatio'].append(fRatio)
    ChannelPowerFrame = pd.DataFrame(ChannelPowerDict)
    Dataframe = Dataframe.merge(ChannelPowerFrame, on='fID', how='left')
    Dataframe.to_csv('MetricsByChannel_' + str(Run) + '.csv', index=False)
    print(Dataframe.head())

    return PowerFrame

def main():
    # Preliminary configuration
    sg.signal(sg.SIGINT, SigintHandler)
    cfg = ReturnConfig('TPCConfig.yaml')
    logging.basicConfig(filename=cfg['Miscellaneous']['LogPath'] + cfg['Miscellaneous']['LogName'], level=logging.DEBUG, filemode='w')
    logging.warning('Logging service has started.')

    # Connect to ROOT file

    FullPower = pd.DataFrame()
    for Run in cfg['Data']['Runs']:
        Map = cfg['Data']['Runs'][Run]
        FileToProcess = Decode(str(Run))
        Data = uproot.open(FileToProcess)
        Events = Data[cfg['Path']['RecoFolder']]
        Power = Analyze(Events, cfg, Run)
        FullPower = FullPower.append(Power)
        Power.to_csv(cfg['Path']['Images']+'Run'+str(Run)+'Power.csv', index=False)

    for Run in cfg['Data']['AnalyzedRuns']:
        Power = pd.read_csv(cfg['Path']['Images']+'Run'+str(Run)+'Power.csv')
        FullPower = FullPower.append(Power)
    FullPower.to_csv('FullPower.csv', index=False)

    #ProcessWE = False
    #ProcessWW = False
    #if cfg['Data']['Runs']['WERun'] != 0: ProcessWE = True
    #if cfg['Data']['Runs']['WWRun'] != 0: ProcessWW = True
    
    #if ProcessWW:
    #    WestFileToProcess = Decode(str(cfg['Data']['WestRun']))        
    #    WestData = uproot.open(WestFileToProcess)
    #    WestEvents = WestData[cfg['Path']['RecoFolder']]
    #    WestPower = AnalyzeHalf(WestEvents, cfg, 'WW')
    #    WestPower.to_csv(cfg['Path']['Images']+'Run'+str(cfg['Data']['Runs']['WestRun'])+'Power_WW.csv', index=False)

    #if ProcessWE:
    #    EastFileToProcess = Decode(str(cfg['Data']['EastRun']))
    #    EastData = uproot.open(EastFileToProcess)
    #    EastEvents = EastData[cfg['Path']['RecoFolder']]
    #    EastPower = AnalyzeHalf(EastEvents, cfg, 'WE')
    #    EastPower.to_csv(cfg['Path']['Images']+'Run'+str(cfg['Data']['Runs']['EastRun'])+'Power_WE.csv', index=False)

    #if ProcessWE and ProcessWW:
    #    PowerFrame = EastPower.append(WestPower)
    #    print(PowerFrame)
    #    PowerFrame.to_csv(cfg['Path']['Images']+'PowerFullAnalysis.csv', index=False)
    #elif ProcessWE:
    #    PowerFrame = EastPower
    #elif ProcessWW:
    #    PowerFrame = WestPower

    # Plot the power in a geographically relevant heatmap. The config field Columns
    # specifies a list of columns to produce a plot for, so we produce a plot for
    # each request column.
    
    for i in range(len(cfg['SVGHeatmap']['Columns'])):
        Tag = cfg['SVGHeatmap']['Columns'][i]
        ZMin = cfg['SVGHeatmap']['ZMin'][i]
        ZMax = cfg['SVGHeatmap']['ZMax'][i]
        BarLabel = cfg['SVGHeatmap']['BarLabel'][i]
        PlotPowerAsHeatmap(PowerFrame,
                           Tag,
                           cfg['SVGHeatmap']['Gradient'],
                           cfg['SVGHeatmap']['SVGBase'],
                           BarLabel,
                           ZMin=ZMin,
                           ZMax=ZMax,
                           EmptyColor=cfg['SVGHeatmap']['EmptyColor'])

if __name__ == "__main__":
    main()
