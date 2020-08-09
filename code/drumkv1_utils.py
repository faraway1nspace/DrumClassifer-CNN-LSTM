# functions for exporting (classsified) samples to a DrumKv1 XML kit
from __future__ import print_function
import librosa
from matplotlib import pyplot as plt
import numpy as np
import librosa.display
import os
import re
import random
from random import uniform
from statistics import mean, stdev,median
import torch
import torch.nn as nn
from torch.autograd import Variable
import lxml.etree
import lxml.builder    
from lxml import etree as ET
import copy

from pytorch_utils import * # pytorch model
from drumclassifier_constants import * # constants
from drumclassifier_utils import * # make code
from drumkv1_constants import * # constants for drumkv1 export

class XMLDrumkv1Maker:   
    """exports samples to an XML format for Drumkv1
    makes one XML kit
    this object doesn't do any classification, merely handles a dictionary of samples and Drumkv1 parameters and writes to XML
    """
    def __init__(self, params_sample=None, params_instrument=None, params_instrument_default_random = None):
        # default parameters per sample
        self.params_sample = {
            ("1","GEN1_REVERSE"):0,
            ("2","GEN1_GROUP"):0,
            ("3","GEN1_COARSE"):0,
            ("4","GEN1_FINE"):0,
            ("5","GEN1_ENVTIME"):0.2,
            ("6","DCF1_CUTOFF"):1,
            ("7","DCF1_RESO"):0,
            ("8","DCF1_TYPE"):0,
            ("9","DCF1_SLOPE"):0,
            ("10","DCF1_ENVELOPE"):1,
            ("11","DCF1_ATTACK"):0,
            ("12","DCF1_DECAY1"):0.5,
            ("13","DCF1_LEVEL2"):0.2,
            ("14","DCF1_DECAY2"):0.5,
            ("15","LFO1_SHAPE"):1,
            ("16","LFO1_WIDTH"):1,
            ("17","LFO1_RATE"):0.5,
            ("18","LFO1_SWEEP"):0,
            ("19","LFO1_PITCH"):0,
            ("20","LFO1_CUTOFF"):0,
            ("21","LFO1_RESO"):0,
            ("22","LFO1_PANNING"):0,
            ("23","LFO1_VOLUME"):0,
            ("24","LFO1_ATTACK"):0,
            ("25","LFO1_DECAY1"):0.5,
            ("26","LFO1_LEVEL2"):0.2,
            ("27","LFO1_DECAY2"):0.5,
            ("28","DCA1_VOLUME"):0.6,
            ("29","DCA1_ATTACK"):0,
            ("30","DCA1_DECAY1"):0.5,
            ("31","DCA1_LEVEL2"):0.2,
            ("32","DCA1_DECAY2"):0.5,
            ("33","OUT1_WIDTH"):0.2,
            ("34","OUT1_PANNING"):0,
            ("35","OUT1_VOLUME"):0.5}
        
        # default parameters for the instrument (e.g., effects)
        self.params_instrument = {
            ("36","DEF1_PITCHBEND"):0.2,
            ("37","DEF1_MODWHEEL"):0.2,
            ("38","DEF1_PRESSURE"):0.2,
            ("39","DEF1_VELOCITY"):0.59,
            ("40","DEF1_CHANNEL"):0,
            ("41","DEF1_NOTEOFF"):1,
            ("42","CHO1_WET"):0,
            ("43","CHO1_DELAY"):0.5,
            ("44","CHO1_FEEDB"):0.5,
            ("45","CHO1_RATE"):0.5,
            ("46","CHO1_MOD"):0.5,
            ("47","FLA1_WET"):0,
            ("48","FLA1_DELAY"):0.5,
            ("49","FLA1_FEEDB"):0.5,
            ("50","FLA1_DAFT"):0,
            ("51","PHA1_WET"):0,
            ("52","PHA1_RATE"):0.5,
            ("53","PHA1_FEEDB"):0.5,
            ("54","PHA1_DEPTH"):0.5,
            ("55","PHA1_DAFT"):0,
            ("56","DEL1_WET"):0,
            ("57","DEL1_DELAY"):0.5,
            ("58","DEL1_FEEDB"):0.5,
            ("59","DEL1_BPM"):180,
            ("60","DEL1_BPMSYNC"):1,
            ("61","DEL1_BPMHOST"):180,
            ("62","REV1_WET"):0,
            ("63","REV1_ROOM"):0.5,
            ("64","REV1_DAMP"):0.5,
            ("65","REV1_FEEDB"):0.5,
            ("66","REV1_WIDTH"):0,
            ("67","DYN1_COMPRESS"):1,
             ("68","DYN1_LIMITER"):0}
        
        if params_sample is not None:
            for (ind_,name_), v in params_sample.items():
                print("adding parameter %s with value %0.3f" %(name_, v))
                self.params_sample[(ind_,name_)] = v
        
        if params_instrument is not None:
            for (ind_,name_), v in params_instrument.items():
                print("adding parameter %s with value %0.3f" %(name_, v))
                self.params_instrument[(ind_,name_)] = v
        
        self.params_random = params_instrument_default_random
        
        # initialize the XML Element builder
        self.E = lxml.builder.ElementMaker()
        
    def init_xml_skeleton(self, save_path, version=None):
        # name of instrument
        self.instrument_name = save_path.split("/")[-1].split(".")[0]
        self.save_path = save_path
        
        # verion
        if version is None:
            version = "0.6.3"
        self.version = str(version)
        
        # make the base XML skeleton
        self.xmlobj = self.E.preset(self.E.elements(), version=self.version, name=self.instrument_name)
        # append all the broad instrument parameters
        self.xmlobj.append(self.E.params()) # append <params>
        # LXML: append all the default parameters
        for (ind_,name_), v in self.params_instrument.items(): # append <param>
            self.xmlobj[1].append(self.E.param(str(v),index=str(ind_), name = name_))
        
        # set defaults
        self.key = 35
        self.keys_assigned = {}
        
    def make_element(self, path_to_sample, index_element=None, do_return=False,do_append=True, params_random=None):
        # update the key
        if index_element is None:
            self.key += 1
            index_element = self.key
            
        if index_element is self.keys_assigned:
            print("Key %d is already assigned" % (index_element))
            raise ValueError("Key %d is already assigned" % (index_element))
        
        # LXML syntax: base of the syntax
        elem_ = self.E.element(
            self.E.sample(path_to_sample, index="0", name = "GEN1_SAMPLE"),
            self.E.params(
                self.E.param(str(index_element),index="0", name = "GEN1_SAMPLE")
                ),
            index= str(index_element)
            )
        
        # LXML: append all the default parameters
        for (ind_,name_), v in self.params_sample.items():
            elem_[1].append(self.E.param(str(v),index=str(ind_), name = name_))
        
        # random parameters:
        if (params_random is None) and (self.params_random is not None):
            params_random = self.params_random
        
        if params_random is not None:
            for (ind_,name_), minmax_ in params_random.items():
                if len(minmax_)!=2:
                    print("random parameter for %s requires a list of two elements, min and max" % name_)
                else:
                    v = uniform(min(minmax_), max(minmax_))
                    # find the corresponding parameter that matches
                    for param_ in elem_[1]:
                        if param_.attrib['name'] == name_:
                            param_.text = str(round(v,4))
        
        # add the key to the assigned keys
        self.keys_assigned[index_element] = path_to_sample
        
        if do_append:
            self.xmlobj[0].append(elem_)
        
        if do_return:
            return elem_
    
    def write(self, path=None):
        """ export """
        if path is None:
            path = self.save_path
        
        # make string
        xmlstring = ET.tostring(self.xmlobj, xml_declaration=False, doctype='<!DOCTYPE drumkv1>', encoding='utf-8', standalone=False, with_tail=False, method='xml', pretty_print=True)
        xmlstring = xmlstring.decode("utf-8")
        
        # remove the header
        m = re.match(r"<.*>", xmlstring)
        xmlstring = xmlstring[(m.span()[1]+1):]
        
        # export
        with open(path,'w') as fcon:
            fcon.write(xmlstring)

class ClassiferPlusKv1Exporter:
    """ 
    Wrapper functionality to serve two objectives:
    - classifies a directory(s) of drum samples, via DrumClassifier object
    - exports a Drumkv1 kit (xml) using the 'make_drumkv1_kit' function and XMLDrumkv1Maker object
    """
    def __init__(self,
                 path_to_model = None,
                 assignment_default = None,
                 params_sample_default=None,
                 params_instrument_default=None,
                 params_sample_random_default = None,
                 arg_DrumClassifier = None,
                 verbose = 1):
        
        self.verbose = verbose
        
        # load the pytorch modelclassifer object
        if path_to_model is None:
            path_to_model = "models/mel_cnn_models/mel_cnn_model_high_v2.model"
        if not os.path.isfile(path_to_model):
            ValueError("path_to_model argument: %s model doest not exist" % path_to_model)
        else:
            if verbose:
                print("loading %s" % path_to_model)
            else:
                pass
        
        self.path_to_model = path_to_model
        # DrumClassifier: handles the pytorch model initialization
        self.drumcl = DrumClassifier(path_to_model=self.path_to_model, verbose = self.verbose)
        
        # collect defaults: assignment of keys (in lieu of any additional instruction by user)
        if assignment_default is None:
           assignment_default =  DEFAULT_ASSIGNMENT
        self.assignment_default = assignment_default
        
        # default parameters for drumkv1 export: 
        if params_sample_default is None:
            params_sample_default= PARAMS_SAMPLE_DEFAULT
        self.params_sample_default = params_sample_default
        
        # default parameters for overall instrument
        if params_instrument_default is None: 
            params_instrument_default=PARAMS_INSTRUMENT_DEFAULT
        self.params_instrument_default = params_instrument_default
        
        # default randomization of panning and width: 
        if params_sample_random_default is None:
            params_sample_random_default = PARAMS_SAMPLE_RANDOM_DEFAULT
        self.params_sample_random_default = params_sample_random_default
        
        # get the default file types from the drumclassifier
        self.file_types = self.drumcl.file_types
        
        # empty container: collecting VERIFIED samples
        self.sample_paths_checked_and_ready = []
    
    def resolve_samples_path_argument(self, samples):
        """ recursive: finds any/all sound files in a directory or path
        checks permissible file-types against
        """
        # is char: check files and/or directories
        if isinstance(samples, str):
            # check file type
            is_dir = os.path.isdir(samples)
            is_file = os.path.isfile(samples)
            if is_file:
                file_type_ = samples.split(".")[-1].lower()
                file_type_check = file_type_ in self.file_types
                if file_type_check:
                    self.sample_paths_checked_and_ready.append(samples)
                    if self.verbose:
                        print("found %s" % samples)
            elif is_dir:
                # add / to end
                curpath = samples
                if curpath[-1]!='/':
                    curpath += '/'
                
                get_paths_in_dir = [curpath+f for f in os.listdir(samples)]
                if len(get_paths_in_dir)>0:
                    if self.verbose:
                        print("found directory %s; traversing for files..." % curpath)
                    self.resolve_samples_path_argument(get_paths_in_dir)
            
            else:
                print("path %s does not exist, check again" % samples)
        
        # list, loop through
        elif isinstance(samples, list):
            for samp_ in samples:
                self.resolve_samples_path_argument(samp_)
        
        else:
            print("'samples' argument must be a list (of paths to samples) or a directory")
    
    def resolve_samples(self, samples):
        """ recursive: finds any/all sound files in a directory or path; 
            returns a list of full-paths to sound files
        """ 
        self.sample_paths_checked_and_ready = []
        # resolve all files in 'samples'
        self.resolve_samples_path_argument(samples)
        if len(self.sample_paths_checked_and_ready)==0:
            ValueError("didn't find any files in samples argument '%s'" %s)
        # reset:
        sample_paths_checked_and_ready = copy.copy(self.sample_paths_checked_and_ready)
        self.sample_paths_checked_and_ready = []
        # r
        return sample_paths_checked_and_ready
    
    def classify_list(self, list_of_samples,format=None):
        if format is None:
            format = 'mat'
        if len(list_of_samples)>0:
            if self.verbose:
                print("starting pytorch classification")
            
            classification_results = self.drumcl.predict_list_of_files(list_of_samples, format=format)
            return classification_results
        else:
            ValueError("empty list for self.drumcl; nothing to classify")
    
    def assign_key_to_class(self, key_map, predictions, class_, keys_):
        """ simple allocates a class to media-keys (based on predictions)"""
        # get prediction vector for this class
        idxcol = np.array([idx for idx,cl in enumerate(predictions['col']) if cl== class_])[0]
        pred_vec = predictions['prob'][:,idxcol]
        if isinstance(keys_, int):
            keys_ = [keys_]
        nkeys_to_assign = len(keys_)
        # highest probability samples (indcies)
        idx_high_prob_samples = np.argsort(-1*pred_vec)[0:nkeys_to_assign]
        # names/paths to high-prob samples
        path_high_prob_samples = [predictions['row'][ix] for ix in idx_high_prob_samples]
        # assign
        for sample_path, key_ in zip(path_high_prob_samples, keys_):
            #key_map[sample_path] = {'key':key_}
            key_map[key_] = {'path':sample_path}
        return key_map
    
    def assign_preds_to_keys(self, predictions, assignment=None, method = None):
        """
        function plural of 'assign_key_to_class'
        takes results from 'classify_list'; and maps to keys
        arguments:
        -'predictions': output from 'classify_list'
        -'method': either 'class' or 'sample'
        -'assignment': a dictionary with keys-as-classes(predicted) and values as the midi-key to assing 
        --this refers to one of two methods to assign (multiple) samples to multiple keys
        ----'class': it will fill each class in 'assignment' with its highest-probability sample
        ----  disadvantage is that the same sample may be the MAP for different classes
        ----'sample': it will assign every sample to its highest probability class  
        ----  disadvantage is that some classes may be unclassified, and thus have no midi-key!
        """
        if method is None:
            method = 'class'
        if not (method == 'sample' or method == 'class'):
            print("'method' argument must be either 'sample' or 'class'. Switching to 'class'")
            method = 'class'
        
        if assignment is None:
            if self.verbose:
                print("using default key-assignment")
            assignment = copy.copy(self.assignment_default)
        
        if method == 'class':
            key_map = {}
            for class_, keys_ in assignment['key_assignment'].items():
                key_map = self.assign_key_to_class(key_map, predictions, class_, keys_)
            
            return key_map
        
        elif method == 'sample':
            ValueError("method not yet implimented.SoRrYYYY! send me email to do")
    
    def export_kit(self,
                   path_to_kit,
                   key_assignment,
                   params_sample=None,
                   params_instrument=None,
                   params_instrument_default_random = None,
                   overwrite=None):
        """ just a wrapper for function: make_drumkv1_kit"""
        if overwrite is None:
            overwrite = False
        
        if os.path.isfile(path_to_kit) and overwrite:
            print("warning: over-writing %s" % path_to_kit)
        
        elif os.path.isfile(path_to_kit) and not overwrite:
            rename_kit_tmp = path_to_kit.replace(".drumkv1", "-1.drumkv1")
            print("warning: %s already exists; exporting to %s" % (path_to_kit, rename_kit_tmp))
            path_to_kit = rename_kit_tmp
        
        if params_sample is None:
            params_sample = self.params_sample_default
        
        if params_instrument is None:
            params_instrument = self.params_instrument_default
        
        if params_instrument_default_random is None:
            params_instrument_default_random = self.params_sample_random_default
        
        # make kit
        make_drumkv1_kit_by_keys(path_to_kit,
                                 midikeys_and_sample_params = key_assignment, 
                                 params_sample=params_sample,
                                 params_instrument=params_instrument,
                                 params_instrument_default_random = params_instrument_default_random,
                                 overwrite=overwrite,
                                 verbose=self.verbose)
        
        #make_drumkv1_kit(path_to_kit = path_to_kit,
        #         sample_paths_and_params = key_assignment,
        #         params_sample=params_sample,
        #         params_instrument=params_instrument,
        #         params_instrument_default_random = params_instrument_default_random,
        #         overwrite=overwrite,
        #         verbose=self.verbose)
        pass
    
    # main function: does prediction and exports to Drumkv1 kit
    def classify_and_make_kit(self, path_to_kit,
                              samples,
                              assignment = None,
                              params_sample=None,
                              params_instrument=None,
                              params_instrument_default_random = None,
                              overwrite=None):
        """ 
        main function: classifies and exports the results as a drumkv1 kit
        args:
         - samples (character, directory): uses all audio files in directory
         - samples (list, paths): uses all audio files referenced in list
         - samples (list, directories): combines all files into one MEGA-kit
        see 'classify_and_make_kits' to build separate kits per directory input
        """
        # check/resolve all samples in argument 'samples'
        list_of_samples = self.resolve_samples(samples)
        # make load samples and run torch model
        predictions = self.classify_list(list_of_samples, format = 'mat')
        # remap predicted-classes to midi-keys
        key_map = self.assign_preds_to_keys(predictions, assignment) #
        # export kit
        self.export_kit(path_to_kit=path_to_kit,
                        key_assignment=key_map,
                        params_sample=params_sample,
                        params_instrument=params_instrument,
                        params_instrument_default_random = params_instrument_default_random,
                        overwrite=overwrite)
        if self.verbose:
            print("DONE CLASSIFICATION AND EXPORT")
    
    # wrapper for classify_and_make_kit
    def classifies_and_make_kits(self, path_to_kits,
                                 directories,
                              assignment = None,
                              params_sample=None,
                              params_instrument=None,
                              params_instrument_default_random = None,
                              overwrite=None):
        """ 
        wraps classify_and_make_kit, works on a list of directories
        args:
         - directories (list): makes a kit for each directory
         - samples (list, paths): uses all audio files referenced in list
         - samples (list, directories): combines all files into one MEGA-kit
         - path_to_kits: base path
        see 'classify_and_make_kits' to build separate kits per directory input
        """
        if not os.path.isdir(path_to_kits):
            raise ValueError("%s does not exist" % path_to_kits)
        else:
            print("exporting kits to %s" % path_to_kits)
        if path_to_kits[-1] != "/":
            path_to_kits += "/"
        
        if isinstance(directories,list):
            # if directory, make a kit out of the directory
            for dir_ in directories:
                # check it's a directory
                if os.path.isdir(dir_):
                    #kit_name = dir_.split("/")[-1]
                    kit_name = [folder for folder in dir_.split("/") if len(folder)>0][-1]
                    path_to_kit = path_to_kits + kit_name + ".drumkv1"
                    print("Making path_to_kit %s from directory %s" % (path_to_kit, dir_))
                    # make kit
                    self.classify_and_make_kit(path_to_kit=path_to_kit,
                                               samples=dir_,
                                               assignment=assignment,
                                               params_sample=params_sample,
                                               params_instrument=params_instrument,
                                               params_instrument_default_random=params_instrument_default_random,
                                               overwrite=overwrite)
                    del kit_name, path_to_kit
                else:
                    print("skipping %s: not a directory or doesn't exist" % dir_)
        elif isinstance(directories, str):
            if os.path.isdir(directories):
                list_of_directories = [os.path.join(directories,dir_) for dir_ in os.listdir(directories) if os.path.isdir(os.path.join(directories,dir_))]
                if len(list_of_directories)>0:
                    if self.verbose:
                        print("traversing %s for other directories to making into kits" % directories)
                        print("found directories: %s" % ";".join(list_of_directories))
                        self.classifies_and_make_kits(path_to_kits,
                                                      list_of_directories,
                                                      assignment,
                                                      params_sample,
                                                      params_instrument,
                                                      params_instrument_default_random,
                                                      overwrite)
                else:
                    print("nothing in %s; exiting" % directories)
            else:
                print("skipping %s: not a directory or doesn't exist" % directories)
        else:
            print("%s must be either a list of directories, or a directory to traverse for directories" % directories)

# export function: 
def make_drumkv1_kit_by_keys(path_to_kit,
                     midikeys_and_sample_params,
                     params_sample=None,
                     params_instrument=None,
                     params_instrument_default_random = None,
                     overwrite=False,
                     verbose=True):
    """
    'make_drumkv1_kit_by_keys' this version exports a kit and is organized by keys. I.e., the main input 'midikeys_and_sample_params' is a dictionary {MIDIKEY:DICT_OF_SAMPLE_PARAMETERS}, where the midikey is one unique 128 key on a midikey board, and the DICT_OF_SAMPLE_PARAMETERS is a dictionary that with keys 'path' that points to a local sample, and other option drumvk1 parameters
    """
    if path_to_kit[-8:].lower() != '.drumkv1':
        print("appending drumkv1 extension to path")
        path_to_kit += ".drumkv1" 
    
    # name
    kit_name = path_to_kit.split("/")[-1].split(".")
    # check if the kit exists
    raise_error = False
    if os.path.isfile(path_to_kit) and (not overwrite):
        raise_error = True
        print("Warning: %s already exists. Choose new name or set 'overwrite' to True; exiting with error" % path_to_kit)
    
    # check the integrity of all the samples
    for _, sample_dict in midikeys_and_sample_params.items():
        sample_path = sample_dict['path']
        if not os.path.isfile(sample_path):
            print("%s does not exist" % sample_path)
            raise_error = True
    
    # raise errors
    if raise_error:
        raise ValueError("see warnings above")
    
    # intialize the XML object
    xmlobject = XMLDrumkv1Maker(params_sample=params_sample,
                                params_instrument_default_random = params_instrument_default_random)
    # initialize
    xmlobject.init_xml_skeleton(save_path=path_to_kit)
    # loop through samples
    for key_, sample_dict in midikeys_and_sample_params.items():
        # path to sample
        sample_path = sample_dict['path']
        # midi key to assign
        key_to_assign = str(key_)
        # some optional random parameters
        if 'params_random' in sample_dict:
            params_random = sample_dict['params_random']
        else:
            params_random = None
        
        # add sample to xmlobject-kit
        xmlobject.make_element(path_to_sample=sample_path, index_element = key_to_assign, do_append=True,do_return=False, params_random = params_random)
        if verbose:
            print("added %s to key %s" % (sample_path, key_to_assign))
    
    # write
    xmlobject.write()
    if verbose:
        print("wrote kit to %s; %d" % (path_to_kit, int(os.path.isfile(path_to_kit))))

# export function
def make_drumkv1_kit_by_sample(path_to_kit,
                     sample_paths_and_params,
                     params_sample=None,
                     params_instrument=None,
                     params_instrument_default_random = None,
                     overwrite=False,
                     verbose=True):
    """
    this takes a dictionary of samples, where the samples are the keys, and the values have parameters and a key assignment. However, key assignment ISN'T necessary, it can be done automatically.
    see 'make_drumkv1_kit_by_key' for a verion that is organized by midi keys
    """
    if path_to_kit[-8:].lower() != '.drumkv1':
        print("appending drumkv1 extension to path")
        path_to_kit += ".drumkv1" 
    
    # name
    kit_name = path_to_kit.split("/")[-1].split(".")
    # check if the kit exists
    raise_error = False
    if os.path.isfile(path_to_kit) and (not overwrite):
        raise_error = True
        print("Warning: %s already exists. Choose new name or set 'overwrite' to True; exiting with error" % path_to_kit)
    
    # check the integrity of all the samples
    for sample_path, sample_dict in sample_paths_and_params.items():
        
        if not os.path.isfile(sample_path):
            print("%s does not exist" % sample_path)
            raise_error = True
    
    # raise errors
    if raise_error:
        raise ValueError("see warnings above")
    
    # intialize the XML object
    xmlobject = XMLDrumkv1Maker(params_sample=params_sample,
                                params_instrument_default_random = params_instrument_default_random)
    # initialize
    xmlobject.init_xml_skeleton(save_path=path_to_kit)
    # loop through samples
    for sample_path, sample_dict in sample_paths_and_params.items():
        if 'params_random' in sample_dict:
            params_random = sample_dict['params_random']
        else:
            params_random = None
        if 'key' in sample_dict:
            key_to_assign = str(sample_dict['key'])
        else:
            key_to_assign = None
        # add sample to xmlobject-kit
        xmlobject.make_element(path_to_sample=sample_path, index_element = key_to_assign, do_append=True,do_return=False, params_random = params_random)
        if verbose:
            print("added %s to key %s" % (sample_path, key_to_assign))
    
    # write
    xmlobject.write()
    if verbose:
        print("wrote kit to %s; %d" % (path_to_kit, int(os.path.isfile(path_to_kit))))

