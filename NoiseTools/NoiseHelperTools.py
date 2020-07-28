import numpy as np
import sys
import os
import yaml
import logging
import glob
import samweb_client

def SigintHandler(signal, frame):
    # This function is meant to handle SIGINT signals sent by the user. For now it only logs
    # the SIGINT and exits.

    logging.error('SIGINT sent. Exiting...')
    print('Received SIGINT. Exiting...')
    sys.exit(0)

def Decode(Run):
    # This function interacts with Samweb to locate the run file for the requested run.
    # Either the run has already been decoded, in which case the path to the decoded file
    # is returned, or the run file needs to be decoded, in which case the function launches
    # a LArSoft job to decode the file. In either case the function returns the path to
    # a decoded file for the requested run.

    logging.debug('[ Decode() ]: Decode() called with Run = ' + Run)

    # Configure the samweb_client object, which serves as a python interface for Samweb.
    # We need to tell it which experiment (ICARUS) the data files are as part of the
    # configuration process. Then we list the data files for the requested run number.
    samweb = samweb_client.SAMWebClient(experiment='icarus')
    Files = np.array(samweb.listFiles('run_number='+Run))
    logging.debug('[ Decode() ]: Files returned by Samweb: \n' + str(Files))
    
    # By default we select the first file for each run, which is facilitated by the boolean
    # mask. In theory a file from each DataLogger could pass the mask, but as of now the only
    # DataLogger being sent to tape is DataLogger1. Then we use Samweb to find the locations
    # of the files and append to the paths the file name. This creates a list with the full
    # path of each of the files.
    Mask = [ '_1_' in x for x in Files ]
    logging.debug('[ Decode() ]: Selected file: ' + str(Files[Mask]))
    Locations = [ samweb.locateFile(x)[0]['full_path'].split(':')[1] for x in Files[Mask] ]
    ToProcess = [ Locations[x] + '/' +  Files[Mask][x] for x in range(len(Files[Mask])) ]
    logging.debug('[ Decode() ]: Full path of file: ' + str(ToProcess))

    # The decoder tends to produce Supplemental files, which we should tidy up.
    if len(glob.glob('Supplemental*.root')) > 0: 
        os.system('rm Supplemental*.root')
        logging.debug('[ Decode() ]: Removing supplemental file.')
    
    # Now we check if there are any already decoded files for the run, and if not we launch a
    # a decoder job on the first file in ToProcess.
    MatchingFiles = glob.glob('*' + Run + '_1_*-decode.root')
    logging.debug('[ Decode() ]: Matching glob for run: ' + str(MatchingFiles))
    if len(MatchingFiles) < 1:
        logging.debug('[ Decode() ]: No decoded file found for run. Begin decode job.')
        os.system('lar -c /icarus/app/users/mueller/NoiseStudies_v08_59_00/workdir/decoder.fcl -n 50 ' + ToProcess[0])
        os.system('rm Supplemental*.root')
    
    # We want to double check that the decoder job was successful. If we find a decoded file,
    # then return the full path. Else tell the user that no file has been found and exit.
    MatchingFiles = glob.glob('*' + Run + '_1_*-decode.root')
    if len(MatchingFiles) > 0:
        logging.debug('[ Decode() ]: Successfully returning file: ' + MatchingFiles[0])
        return MatchingFiles[0]
    else:
        logging.debug('[ Decode() ]: No matching file found. Returning None.')
        print('No matching files.')
        sys.exit(0)

def ReturnConfig(ConfigFile):
    # A YAML formatted file is used to store the various configuration settings for the
    # analysis code. This functions loads it from a file and then uses the yaml package
    # to parse the settings into a dictionary of configurations.
    
    with open(ConfigFile, 'r') as Config:
        cfg = Config.read()
        return yaml.load(cfg, Loader=yaml.Loader)
