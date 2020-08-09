from __future__ import print_function
import librosa
from matplotlib import pyplot as plt
import numpy as np
import librosa.display
import os
import re
import random
from statistics import mean, stdev,median
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorch_utils import *
from drumclassifier_constants import * # constants

def transform(y,sr,hop_length):
    """ convert audio-raws format in mel-coefficients"""
    M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length)
    log_M = librosa.amplitude_to_db(M, ref=np.max)
    M_std = (log_M+80)/80
    return M_std

class DrumClassifier:
    """ main object; loads pytorch model and has functions for various sorts of classification
    The most user-friendly function would be 'predict_list_of_files', which classifies a list of files
    or 'predict_proba_directory' which classifies all files in a directory
    Arguments.
    path_to_model: loads trained pytorch model
    file_types: acceptable file-types (['aif', 'flac', 'wav', 'ogg','WAV','AIF','FLAC','OGG'])
    hop_length: 256
    """
    def __init__(self, path_to_model=None, file_types = None, hop_length = None, clip=None, bs=None, maxseqlen = None, minseqlen=None,pad=None, int2class = None, verbose = 1):
        if path_to_model is None:
            path_to_model = "../models/mel_cnn_models/mel_cnn_model_high_v2.model"
        if hop_length is None:
            hop_length = 256
        # default hop_length
        self.hop_length = hop_length
        # default clip size
        if clip is None:
            clip = 600
        self.clip = clip
        # default files_types
        if file_types is None:
            file_types = ['aif', 'flac', 'wav', 'ogg','WAV','AIF','FLAC','OGG']
        self.file_types = file_types
        if bs is None:
            bs = 12
        self.bs = bs
        if maxseqlen is None:
            maxseqlen = 300
        self.maxseqlen = maxseqlen
        if minseqlen is None:
            minseqlen=17
        self.minseqlen = minseqlen
        if pad is None:
            pad=0
        self.pad = pad
        # integer to class
        if int2class is None:
            int2class = {0: 'whis', 1: 'met', 2: 'cow', 3: 'taik', 4: 'clav', 5: 'hhc', 6: 'cah', 7: 'tim', 8: 'tab', 9: 'kick', 10: 'rid', 11: 'bass', 12: 'thd', 13: 'hho', 14: 'rim', 15: 'shk', 16: 'stick', 17: 'tom', 18: 'clp', 19: 'wood', 20: 'cong', 21: 'snr', 22: 'tri', 23: 'vib', 24: 'cym', 25: 'gui', 26: 'fx', 27: 'tomf', 28: 'ped', 29: 'cui'}
        self.int2class = int2class
        self.class2int = {v:k for k,v in int2class.items()}
        assert os.path.isfile(path_to_model)
        
        # get saved model parameters
        modstate = torch.load(path_to_model)
        
        # initialize the model
        self.net = Classifier(HyperParams(modstate['params']))
        
        # populate model parameters
        self.net.load_state_dict(modstate['model_state_dict'])
        self.net.eval()
        
        # verbose
        self.verbose = verbose
    
    def transform(self, y,sr):
        M = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=self.hop_length)
        log_M = librosa.amplitude_to_db(M, ref=np.max)
        M_std = (log_M+80)/80
        
        # clip
        if M_std.shape[-1]>self.clip:
            M_std = M_std[:,0:self.clip]
        
        return M_std
    
    def check_file_type(self,filename):
        """ check that file is acceptable"""
        suffix = filename.split("/")[-1].split(".")[-1]
        do_return = suffix.lower() in self.file_types
        if self.verbose:
            if not do_return:
                print("file %s not an acceptable file format")
        return do_return
    
    def load_file(self, filename): 
        # load audio file: librosa,
        if self.check_file_type(filename):
            try:
                y, sr = librosa.load(filename)
                return y,sr
            except:
                print("failed loading %s" % filename)
                return None,None
        else:
            return None,None
    
    def load_file_and_transform(self, filename):
        """ load file, and transform to MEL coefficients"""
        y,sr = self.load_file(filename)
        if y is not None:
            return self.transform(y,sr)
        else:
            return None
    
    def load_list_of_files(self,list_of_files):
        """ takes a list of files, return transform spectrograms as numpy arrays"""
        dProcessed_spectrograms = {}
        for filename in list_of_files:
            M = self.load_file_and_transform(filename)
            if M is not None:
                dProcessed_spectrograms[filename] = M
        return dProcessed_spectrograms
    
    def to_tensor(self, M):
        return torch.from_numpy(batch_y).float()
    
    def data_loader_iterator(self,dict_of_numpy_arrays):
        size = len(dict_of_numpy_arrays)
        keys_ = list(dict_of_numpy_arrays.keys())
        xdim = dict_of_numpy_arrays[keys_[0]].shape[0]
        # calculate number of iterations
        num_iterations = max(1,round(size/self.bs))
        o = np.arange(size)
        for i in range(num_iterations):
            if i == (num_iterations-1):
                batch_order = o[(i*self.bs):]
            else:
                batch_order = o[(i*self.bs):((i+1)*self.bs)]
            
            batch_keys = [keys_[b] for b in batch_order]
            batch_x1 = [dict_of_numpy_arrays[b] for b in batch_keys]
            # maximum length
            maxlength_natural = max([x.shape[1] for x in batch_x1])
            maxlen = max(self.minseqlen,min(self.maxseqlen, maxlength_natural))
            # feature length
            # container
            batchxseq = self.pad*np.ones((len(batch_order), maxlen, xdim))
            for j in range(len(batch_order)):
                curlen = min(batch_x1[j].shape[1], maxlen)
                batchxseq[j][:curlen] = batch_x1[j][:,:curlen].T
            
            batchxseq = torch.from_numpy(batchxseq)
            #batch_y, batchxseq = Variable(batch_y), Variable(batchxseq)
            yield batchxseq.float(), batch_keys
    
    def predict_proba(self, dict_of_numpy_arrays):
        """ input: dictionary of mel-transformed coefficients
            output: a dictionary: {filename: probability simplex}"""
        self.net.eval()
        size = len(dict_of_numpy_arrays)
        # data loader
        data_iterator = self.data_loader_iterator(dict_of_numpy_arrays)
        dPrediction_results = {}
        for i in range(max(1,round(size/self.bs))):
            bx,blabels = next(data_iterator)
            pred = self.net(bx)
            pred = torch.softmax(pred,axis=1)
            pred = pred.detach().numpy()
            dPrediction_results.update({label:p for label,p in zip(blabels,pred)})
            del pred,bx,blabels
        return dPrediction_results
    
    def predict_proba_matrix(self, dict_of_numpy_arrays):   #
        """ input: dictionary of mel-transformed coefficients
            output: matrix(proabilities), 'row'=filenames (list), and 'col'=classes (list)"""
        dPredict_proba = self.predict_proba(dict_of_numpy_arrays)
        return {'prob':np.array([x.T for x in dPredict_proba.values()]),
                'row':list(dPredict_proba.keys()),
                'col':list(self.int2class.values())}
        
    def predict(self, dict_of_numpy_arrays):
        """ returns a dictionary {file:integer} intergers can be mapoed with self.int2class """ 
        dPredict_proba = self.predict_proba(dict_of_numpy_arrays)
        # change to integers
        return {file:np.argmax(pred) for file,pred in dPredict_proba.items()}
    
    def label(self, dict_of_numpy_arrays):
        """ returns a dictionary {file:label} """
        dPredict_proba = self.predict_proba(dict_of_numpy_arrays)
        # change to integers
        return {file:self.int2class[np.argmax(pred)] for file,pred in dPredict_proba.items()}
    
    def rank(self,dict_of_numpy_arrays,cutoff = None):
        """ returns the most likely classes, ranked"""
        dPredict_proba = self.predict_proba(dict_of_numpy_arrays)
        # change to integers
        return {file:[self.int2class[k] for k in np.argsort(-1*pred)] for file,pred in dPredict_proba.items()}
    
    def dictproba(self, dict_of_numpy_arrays):
        """ returns a dictionary (keys = files), where key values is a dictionary
        each (sub) dictionary has the class-label as key, and the class-probability as value
        """
        dPredict_proba = self.predict_proba(dict_of_numpy_arrays)
        return {file:{class_:pred[int_] for class_,int_ in self.class2int.items()} for file,pred in dPredict_proba.items()}
    
    def predict_list_of_files(self, list_of_files, format=None):
        """ 
        Args:
        'list of files': pythonlist of paths to samples
        format: 'mat': returns a matrix of probabilties and file names as list
        format: 'rank': returns a dictionary, every class is assigned a file
        format: 'label': returns a dictionary, every sample is assigned its highest-prob class
        format: other: returns the raw prediction probabilities per sample
        """
        if self.verbose:
            print("loading files:%s" % ",".join(list_of_files))
        
        #  transform audio files into matrices for torch model
        dM = self.load_list_of_files(list_of_files)
        
        # cases:
        if 'label' in format.lower():
            return self.label(dM)
        elif 'rank' in format.lower():
            return self.rank(dM)
        elif 'mat' in format.lower():
            prediction_dict = self.predict_proba_matrix(dM)
            return prediction_dict # matrix of probs, files, and classes
        elif (format.lower() == 'dictproba') or (format.lower() == 'prob'):
            return self.dictproba(dM)
        elif format is None:
            prediction_dict = self.predict_proba_matrix(dM)
            return prediction_dict # matrix of probs, files, and classes
        else:
            return self.predict_proba(dM)
    
    def predict_proba_directory(self, dir, format=None):
        """ 
        predict on all files in directory
        wrapper function for 
        """
        if not os.path.isdir(dir):
            raise ValueError("%s is not a valid directory" % dir)
        else:
            files_to_load = [os.path.join(dir,f) for f in os.listdir(dir) if (f.split(".")[-1].lower() in self.file_types)]
            return self.predict_list_of_files(files_to_load, format)
