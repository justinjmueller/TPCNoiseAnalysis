import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from NoiseCalcTools import MeanPower

def PlotRMS(Frame, MiniCrate, Path, Suffix=''):
    # This function creates a simple plot of the RMS as a function of the 'local' (0-575)
    # channel number. Both the full RMS and the RMS after coherent noise removal are
    # plotted. The resulting plot is written as a png to the specified directory.

    # The typical number of channels is 576, but there are a select few mini-crates with
    # 512. Also, we need to create the figure and axes for the plot. In this case we use
    # two subplots stacked vertically with a shared x-axis.
    ChannelRange = np.array(range(0, 576))
    fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    
    # We want to make sure that the dataframe is sorted by the 'local' channel number.
    # Additionally, we need RMS for each channel as 1D arrays (both full and coherent
    # removed).
    SortedFrame = Frame.sort_values(by=['fChannel'])
    RMSRaw = SortedFrame.fRMS.to_numpy()
    RMSUncor = SortedFrame.fUnRMS.to_numpy()

    # We then create scatter plots on the appropriate axis, taking care to adjust the
    # channel range to account for some mini-crates having 512 channels.
    axs[0].scatter(ChannelRange[0:len(RMSRaw)], RMSRaw, s=2, c='b')
    axs[1].scatter(ChannelRange[0:len(RMSUncor)], RMSUncor, s=2, c='b')

    # The usual graph configuring...
    axs[0].set_title('Mean RMS')
    axs[0].set_ylabel('RMS [ADC]')
    axs[0].set_xlim(xmin=0, xmax=576)
    axs[0].set_ylim(ymin=0, ymax=25)
    axs[1].set_title('RMS After Coherent Noise Removal')
    axs[1].set_ylabel('RMS [ADC]')
    axs[1].set_ylim(ymin=0, ymax=25)
    axs[1].set_xlabel('Channel Number')

    # Save the figure as a png using the specified path, the mini-crate name, and any
    # supplied suffix.
    plt.savefig(Path + 'RMS_' + MiniCrate + Suffix + '.png')

def PlotPower(Frequency, PowerRaw, PowerUncor, Frame, MiniCrate, Path, Suffix=''):
    # This function creates a simple plot of the power spectrum as a function of the 'local'
    # (0-575) channel number. Both the full noise spectrum and the spectrum after coherent
    # noise removal are plotted. The resulting plot is written as a png to the specified
    # directory.

    # The typical number of channels is 576, but there are a select few mini-crates with
    # 512. Also, we need to create the figure and axes for the plot. In this case we use
    # two subplots stacked vertically with a shared x-axis.
    ChannelRange = np.array(range(0, 576))
    fig, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    
    # We have the power spectrum per channel and need to reduce this down to to the mean
    # power spectrum for the requested mini-crate. This is done by the MeanPower()
    # function.
    PowerRaw_Selected = MeanPower(PowerRaw, Frame, MiniCrate)
    PowerUncor_Selected = MeanPower(PowerUncor, Frame, MiniCrate)

    # Now we create the two scatter plots, taking care to scale the frequency array to a
    # more appropriate 'kHz' unit.
    axs[0].scatter(1000*Frequency, PowerRaw_Selected, s=2, c='b')
    axs[1].scatter(1000*Frequency, PowerUncor_Selected, s=2, c='b')

    # The usual graph configuring...
    axs[0].set_ylim(ymin=0.1, ymax=10000)
    axs[0].set_title('Raw Power Spectrum')
    axs[0].set_yscale('log')
    axs[0].set_xlim(xmin=0, xmax=800)
    axs[1].set_ylim(ymin=0.1, ymax=10000)
    axs[1].set_title('Power Spectrum After Coherent Noise Removal')
    axs[1].set_yscale('log')

    # Save the figure as a png using the specified path, the mini-crate name, and any
    # supplied suffix.
    plt.savefig(Path + 'Power_' + MiniCrate + Suffix + '.png')

def PlotPowerAsHeatmap(PowerFrame, Gradient, SVGBase, ZMin=0, ZMax=10000, EmptyColor='255,255,255'):
    # This function creates a heatmap style plot of the peak powers for each mini-crate. A          
    # SVG graphic of the geographic layout of the TPC mini-crates is used as the base for           
    # this plot. The base contains some helpful 'tags' that allow me to substitute the              
    # proper values for the color of the mini-crate, the gradient of the legend, and any            
    # labels. It's a bit gimmicky, but a geographic layout is what we're after here.                

    # First we focus our attention on the legend. We use matplotlib to access predefined            
    # gradients, which saves a lot of trouble in making our own. Then we create an empty            
    # dictionary to be used as a map for tags and the value they need to be replaced by.            
    # The first thing we add to the dictionary are the colors that are used as points for           
    # interpolation to a full gradient. Simultaneously we also set each of the labels for           
    # the ticks in the legend.                                                                      
    nTicks = 6
    CMap = cm.get_cmap(Gradient)
    LegendTicks = [ x * (ZMax-ZMin)/5 for x in range(nTicks) ]
    Changes = dict()
    for tick in range(nTicks):
        Changes['$col' + str(tick+1) + '$'] = colors.to_hex(CMap(0.2*tick))
        Changes['$val' + str(tick+1) + '$'] = str(LegendTicks[tick])

    # Now we configure the changes that need to be made to the color of each mini-crate. A
    # lambda function is used to normalize to the requested ZMin and ZMax. The location             
    # where the color is set for each mini-crate in the SVG base file is tagged with the            
    # mini-crate name (e.g. $WE05$). Therefore we need only swap this tag with the color            
    # in rgb notation (e.g. rgb(0,0,0)). Of course it may be the case that not all mini-            
    # crates are represented in the dataframe, so we need to set the remaining mini-crates          
    # as well to some neutral color. To simplify this a little, we can set the default first        
    # then override with the proper color if applicable. I define another lambda called rgb         
    # to format the result from CMap appropriately.
    norm = lambda x : ( ( x - ZMin ) / (ZMax - ZMin) )
    rgb = lambda x : str(tuple( 255*np.array(x[0:3]) ))
    MiniCrateList = ['EE01B', 'EE01M', 'EE01T', 'EE02', 'EE03', 'EE04',
                     'EE05', 'EE06', 'EE07', 'EE08', 'EE09', 'EE10',
                     'EE11', 'EE12', 'EE13', 'EE14', 'EE15', 'EE16',
                     'EE17', 'EE18', 'EE19', 'EE20B', 'EE20M', 'EE20T',
                     'EW01B', 'EW01M', 'EW01T', 'EW02', 'EW03', 'EW04',
                     'EW05', 'EW06', 'EW07', 'EW08', 'EW09', 'EW10',
                     'EW11', 'EW12', 'EW13', 'EW14', 'EW15', 'EW16',
                     'EW17', 'EW18', 'EW19', 'EW20B', 'EW20M', 'EW20T',
                     'WE01B', 'WE01M', 'WE01T', 'WE02', 'WE03', 'WE04',
                     'WE05', 'WE06', 'WE07', 'WE08', 'WE09', 'WE10',
                     'WE11', 'WE12', 'WE13', 'WE14', 'WE15', 'WE16',
                     'WE17', 'WE18', 'WE19', 'WE20B', 'WE20M', 'WE20T',
                     'WW01B', 'WW01M', 'WW01T', 'WW02', 'WW03', 'WW04',
                     'WW05', 'WW06', 'WW07', 'WW08', 'WW09', 'WW10',
                     'WW11', 'WW12', 'WW13', 'WW14', 'WW15', 'WW16',
                     'WW17', 'WW18', 'WW19', 'WW20B', 'WW20M', 'WW20T']
    for MiniCrate in MiniCrateList:
        Changes['$' + MiniCrate + '$'] = 'rgb(' + EmptyColor + ')'
    for index, row in PowerFrame.iterrows():
        Changes['$' + row['fCrate'] + '$'] = 'rgb' + rgb(CMap(norm(row['fPow'])))

    # Now we have all of the changes defined that we need and can proceed with implementing         
    # them. Essentially we only need to read the template out as a string, replace the              
    # tags with the appropriate values as defined by the Changes dictionary, then write the         
    # string to a new file.                                                                         
    with open(SVGBase, 'r') as SVGFile:
        SVG = SVGFile.read()
        for tag in Changes.keys():
            SVG = SVG.replace(tag, Changes[tag])
    OutFile = open('ModSVG.svg', 'w')
    OutFile.write(SVG)
    OutFile.close()
