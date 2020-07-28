import numpy as np
import sys
import logging
import pandas as pd
sys.path.insert(0, '/icarus/app/users/mueller/NoiseStudies/workdir/hdbClient')
from DataLoader3 import DataQuery

def MapChannel(Channel):
    # Unfortunately there is a lot of gymnastics necessary to navigate the hardware
    # database tables. The information we need to create a full channel map is
    # spread across three separate tables.
    
    URL = 'https://dbdata0vm.fnal.gov:9443/QE/hw/app/SQ/query' # The web frontend for the db.
    DATABASE = 'icarus_hardware_dev'                           # Database name.
    DAQCHANNELS = 'daq_channels'                               # Table for DAQ channels.
    READOUTBOARDS = 'readout_boards'                           # Table for readout boards.
    FLANGES = 'flanges'                                        # Table for flanges.

    # The DataQuery class is specifically designed for Fermilab experiment hardware databases
    # and acts as an interface between a hardware database and the user. The DataQuery class
    # requires the URL for the web frontend of the database. The following lines set this
    # object up and then chain together a series of queries to map a DAQ channel to the host
    # mini-crate.
    Query = DataQuery(URL)
    BoardID, BoardSlot, ChannelOnBoard = Query.query(
        DATABASE,
        DAQCHANNELS,
        'readout_board_id, readout_board_slot, channel_number',
        'channel_id:eq:'+str(Channel))[0].split(',')
    FlangeID = Query.query(
        DATABASE,
        READOUTBOARDS,
        'flange_id',
        'readout_board_id:eq:'+BoardID)[0]
    MiniCrate = Query.query(
        DATABASE,
        FLANGES,
        'flange_pos_at_chimney',
        'flange_id:eq:'+FlangeID)[0]
    
    # The result of this gymnastics is the name of the MiniCrate (e.g. 'WE05'), and what 
    # could be considered the 'local' channel number for the DAQ channel. This 'local'
    # channel number refers only to its position on the MiniCrate (0-575). There are 64
    # wires connected to each board, so we take the slot number * 64 with the channel
    # number on the board (0-63) as a further offset.
    return MiniCrate, int(BoardSlot)*64 + int(ChannelOnBoard)

def BuildMapDataFrame(ChannelList, Name='ChannelMap'):
    # Unfortunately it is far too slow to query the hardware database each time we need
    # to map a channel. It is much faster to create a map once and then dump the results
    # to a file for future lookups. This is accomplished through a Pandas dataframe,
    # which keeps the information organized and easy to manipulate.
    
    logging.debug('[ BuildMapDataFrame() ]: BuildMapDataFrame() called with Name = ' + Name)
    
    fMap = {'fID': [], 'fChannel': [], 'fCrate': []} # Used for easy dataframe construction
    
    logging.debug('[ BuildMapDataFrame() ]: Beginning map construction.')
    
    # Now we loop through each channel in the list of channels. For each channel, we
    # map it to a MiniCrate (e.g. formatted as 'WE05') and a 'local' (0-575) channel
    # number. These are stored in the previously constructed dictionary with the fID
    # column corresponding to DAQ channel number, fChannel to the 'local' channel
    # number, and fCrate to the name of the mini-crate formatted as a string (e.g. 'WE05').
    print('Beginning map construction')
    for channel in ChannelList:
        MiniCrate, ChannelOnCrate = MapChannel(channel)
        fMap['fID'].append(channel)
        fMap['fChannel'].append(ChannelOnCrate)
        fMap['fCrate'].append(MiniCrate)
    logging.debug('[ BuildMapDataFrame() ]: Finished map construction.')
    print('Finished map construction')

    # Now we construct a Pandas dataframe and dump it to a file in order to have access
    # to the map much quicker later.
    Dataframe = pd.DataFrame(fMap)
    Dataframe.to_csv(Name+'.csv', index=False)
    logging.debug('[ BuildMapDataFrame() ]: Finished writing map. Exiting BuildMapDataFrame().')

