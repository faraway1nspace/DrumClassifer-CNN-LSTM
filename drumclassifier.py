from __future__ import print_function
import os
import sys
import librosa
import numpy as np
import librosa.display
import re
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
sys.path.append("code/")

# export csv
def export_csv_report(results, filename, verbose=None):
    """
    this write's the prediction from DrumClassifier to a csv file
    Arguments:
     - 'filename' is the output file
     - 'results' is the output from DrumClassifier 'predict_list_of_files' function
    """
    if verbose is None:
        verbose = False
    
    import csv
    
    # check length of 
    
    # writing to csv file  
    with open(filename, 'w') as csvfile:  
        # creating a csv writer object  
        csvwriter = csv.writer(csvfile)
        
        # loop through the results
        counter = -1
        for path_to_file, res in results.items():
            counter+=1
            if isinstance(res,list):
                row_output = res # output
            if isinstance(res, str):
                row_output = [res]
            elif isinstance(res,dict):
                if counter ==0:
                    # if dictionary, write keys as columns headings
                    csvwriter.writerow(["file"]+list(res.keys()))
                # output
                row_output = [str(o) for o in res.values()]
            
            # write to file
            csvwriter.writerow([path_to_file]+row_output)
    
    if verbose:
        print("wrote csv output to %s" % filename)

# main function
def main():
    """
    Either writes to a file (argument 'output') or print results to screen
    Inputs: you can either supply:
     - list of path/to/audio-files (argument '-l') 
     - path/to/directory, such that all files in the directory will be classified (argument '-d')
    'type_output' '-t': allows different types of predicton-outputs, such as probabilities, labels, etc.
     - i) 'label': returns a dictionary with {k:v} as {path_to_file:label (highest-probability classes)}; 
     - ii) 'label': returns a dictionary with {k:v} as {path_to_file:[list of labels orded by probability]; 
     - iii) 'prob': returns a dictionary with {k:v} as {path_to_file:dictionary of class-probabilities}}
    """
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # input is a list of paths to audio-files
    parser.add_argument('-l','--list', nargs='+', help='list of audio files, supplied as comma-delimited sequence', required=False)
    # input is a directory (which is traversed for audio files)
    parser.add_argument('-d','--directory', type=str, help='path to directory with audio files,must be surround with quotations.',  required=False)
    # verbose or not
    parser.add_argument('-o','--output', default=None, type=str, help="optionally output csv", required=False)
    # output type: label, probabilities, or ranks
    parser.add_argument('-t','--type_output', default='label', help="Options: i) 'label': returns a dictionary with {k:v} as {path_to_file:label (highest-probability classes)}; ii) 'label': returns a dictionary with {k:v} as {path_to_file:[list of labels orded by probability]; iii) 'prob': returns a dictionary with {k:v} as {path_to_file:dictionary of class-probabilities}}",required=False)
    # write output to file
    parser.add_argument('-v','--verbose', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), help="verbose", required=False)
    args = parser.parse_args()
    
    verbose = args.verbose
    
    if verbose:
        print("loading DrumClassifier pytorch model")

    # check type output
    type_output = args.type_output
    if type_output not in ['prob', 'rank', 'label', 'mat']:
        raise ValueError()
    
    # load main classes (pytorch model and classifier objections)
    # pytorch CNN-LSTM model        
    from pytorch_utils import DataLoader, make_pos_encodings, HyperParams, Classifier
    
    # constants 
    from drumclassifier_constants import INSTRUMENT_NAMES
    
    # main code
    from drumclassifier_utils import transform, DrumClassifier
    
    # make main Classification object
    drumcl = DrumClassifier(path_to_model= "models/mel_cnn_models/mel_cnn_model_high_v2.model", file_types = None, hop_length = None, clip=None, bs=None, maxseqlen = None, minseqlen=None,pad=None, verbose = verbose)
    
    if args.directory is None and args.list is None:
        raise ValueError("need to specify either argument '--list' or '--directory'")
    
    results = None
    
    # if input is a directory
    if args.directory is not None:
        print(type(args.directory))
        print(args.directory)
        dir_to_audiofiles = str(args.directory)
        if verbose:
            print("loading audio-files in %s" % dir_to_audiofiles)
            print(os.path.isdir(dir_to_audiofiles))
            print(os.listdir(dir_to_audiofiles))
        
        results = drumcl.predict_proba_directory(dir_to_audiofiles, type_output)
    
    # if input is a list
    if args.list is not None:
        if verbose:
            print("loading audio-files: %s" % str(args.list))
        
        # loop through files
        parsed_list_of_potential_files = args.list[0].split(",")
        list_of_files = []
        for path_to_file in parsed_list_of_potential_files:
            if os.path.isfile(path_to_file):
                list_of_files.append(path_to_file)
            else:
                if verbose:
                    print("%s does not exist" % path_to_file)
        if verbose:
            print("begin classification process with DrumClassifier object")
        
        # send list of files to DrumClassifier object
        results = drumcl.predict_list_of_files(list_of_files, type_output)
    
    if verbose:
        print("output is of type %s and length %d" % (str(type(results)), len(results)))
    
    # make csv report
    if args.output is not None:
        export_csv_report(results, args.output, verbose=verbose)
    else:
        print(results)

if __name__ == '__main__':
    main()

# python3.6 drumclassifier.py -l demo_sound_files/snr/snare03.ogg,demo_sound_files/kick/bassdrum02.ogg -v 1 -o label
#python3.6 drumclassifier.py -l demo_sound_files/kick/kick01.ogg,demo_sound_files/hhc/hihat_closed02.ogg,demo_sound_files/hho/hihat_opened01.ogg,demo_sound_files/snr/snare03.ogg -v 1 -o label
