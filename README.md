# DrumClassifier CNN-LSTM (pytorch)

Classify your percussion instruments using a CNN-LSTM model.

This is a deep-learning solution to help organizing my drum-sample collection (audio-files like kicks, snares, cymbals, claps, high-hats, etc.). Searching for the perfect-sounding snare during music production really kills my creativity, so I build this application (and associated tools) to auto-build some custom drumkits.

If you are interested in using this in your work/project, I am open to collaboration and development. My interests:
 - integrate into a DAW as a better-file-browser (like a VST or LV2 plugin)
 - visualize sampels according to their CNN-LSTM latent representations (like a NN-PCA)

### Purpose:
- classify drum-samples (.wav, flac, ogg, aiff) into 30 percussion classes
- (optional) auto-build drumkits based on predicted classes and export to [DrumKV1 kit](https://drumkv1.sourceforge.io/) (a lv2 plugin)
- (future) explore audio-samples according to latent dimensions

### Method:
- python/pytorch based
- a CNN-LSTM 3-layer neural network classifier
- uses MEL-frequency coefficients as the input

### Dependencies
- librosa
- pytorch
- numpy

### Drumkits
One use-case is to auto-organize drumkits in such a way that facilitates a consistent relationship between midi-keys and instruments (i.e., 36 is a snare is a snare is a snare, always). The `drumkv1_utils` functionality will classify drum-samples and arrange them according to a prescribed midi-keys layout (the drumkv1 format). Therefore, while I am exploring drumkits during music production, I can swap among drumkits without scrambling the relationship between midi-keys and the instruments. For example check out these three drumkits built with the Classifier and exported-to-drumkv1.

*play [Cheetah MD16 Drumkit auto-arranged](https://raw.githubusercontent.com/faraway1nspace/DrumClassifer-CNN-LSTM/master/demo_sound_files/riffs/drum-roll2_berklee.mp3)*

*compare to [random Berklee industrial noises](https://raw.githubusercontent.com/faraway1nspace/DrumClassifer-CNN-LSTM/master/demo_sound_files/riffs/drumroll1.mp3)*

The latter consists of random industrial noises have been *forced* into an arrangement that is reminiscient of a conventional drumkit. I find this a fun way to creatively assemble stranges noises into coherent drumkits and inspire new creations.


# Example usage
### command line examples

Let's classify the four demo files in the `demo_sound_files` directory (kick-drum, high-hat closed and open, and a snare
```
./demo_sound_files/kick/kick01.ogg
./demo_sound_files/hhc/hihat_closed02.ogg
./demo_sound_files/hho/hihat_opened01.ogg
./demo_sound_files/snr/snare03.ogg
```

*cmd usage, with input as a list of paths-to-audiofiles (`-l` argument):*
```
python3.6 drumclassifier.py -l demo_sound_files/kick/kick01.ogg,demo_sound_files/hhc/hihat_closed02.ogg,demo_sound_files/hho/hihat_opened01.ogg,demo_sound_files/snr/snare03.ogg
```
output: `{'demo_sound_files/kick/kick01.ogg': 'kick', 'demo_sound_files/hhc/hihat_closed02.ogg': 'hhc', 'demo_sound_files/hho/hihat_opened01.ogg': 'hho', 'demo_sound_files/snr/snare03.ogg': 'snr'}`
The output returns a dictionary with keys being the path/to/audiofile, and the values being the class of percussive instrument.

We can also classify a whole directory of files, with the easier command...
*cmd usage (input is a directory `-d` argument, classifies all files in directory):*
```
python3.6 drumclassifier.py -d "/path/to/directory/" # notice the quotes around path
```

We can get the full probability vectors and save them to a local CSV for further processing/organizing. Here, we use the `-t type_output` argument and set to `prob` (probabilities), as well as setting the `-o` output_file argument to `/tmp/classification_results.csv`

*cmd usage (export to .csv)*:
```
python3.6 drumclassifier.py -l demo_sound_files/kick/kick01.ogg,demo_sound_files/hhc/hihat_closed02.ogg,demo_sound_files/hho/hihat_opened01.ogg,demo_sound_files/snr/snare03.ogg -o /tmp/classification_results.csv -t prob
```
This creates a CSV file, where each row is an audio-file, and each column is a instrument-class, each element is the probability that the audio-file belongs to that class

## Python usage

For more control, you can play with the methods in an interactive python session

```python
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

# load main classes (pytorch model and classifier objections)
# pytorch CNN-LSTM model        
from pytorch_utils import DataLoader, make_pos_encodings, HyperParams, Classifier
    
# constants 
from drumclassifier_constants import INSTRUMENT_NAMES
    
# main code
from drumclassifier_utils import transform, DrumClassifier
    
# make main Classification object
drumcl = DrumClassifier(path_to_model= "models/mel_cnn_models/mel_cnn_model_high_v2.model")

# classify all audio files within a directory, return dictionary of probabilities
results = drumcl.predict_proba_directory("path/to/files/", format = "prob")
```

# Model

The core of the classifier is CNN-LSTM.
- CNN: a stack of three filters, running over the MEL-spectrum and time-dimension. 
- LSTM: collects the 1-D outputs of the CNN and runs them through a final RNN. This allows for time-varying MEL-spectrograms (rather than for artificially fixed-time-length audio files).

### Model in pytorch:
see the file `models/mel_cnn_models/mel_cnn_model_high_v2.model`

```python
class Classifier(nn.Module):   
    def __init__(self,params):
        super(Classifier, self).__init__()
        
        # convolution
        self.convl11 = nn.Sequential(nn.Dropout(params.dropout_11),
                        nn.Conv2d(params.n_channels, params.num_filters_11, kernel_size=params.kerne[l_size_11, stride=params.stride_11),
                        nn.BatchNorm2d(params.num_filters_11),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=params.maxpool_size11, stride=params.maxpool_stride11))
        
        # conv2
        self.convl12 = nn.Sequential(nn.Dropout(params.dropout_12),
                        nn.Conv2d(params.num_filters_11, params.num_filters_12, kernel_size=params.kernel_size_12, stride=params.stride_12),
                        nn.BatchNorm2d(params.num_filters_12),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=params.maxpool_size12, stride=params.maxpool_stride12))
        
        # conv3
        self.convl13 = nn.Sequential(nn.Dropout(params.dropout_13),
                        nn.Conv2d(params.num_filters_12, params.num_filters_13, kernel_size=params.kernel_size_13, stride=params.stride_13),
                        nn.BatchNorm2d(params.num_filters_13),
                        nn.ReLU())
        
        # LSTM
        self.lstm = nn.LSTM(input_size=params.num_filters_13, hidden_size=params.lstm_dim,batch_first=True, bidirectional = True)
        
        # final 2 layer MLP
        self.fc1_dropout = nn.Dropout(params.fc1_dropout)
        self.fc1 = nn.Linear(params.lstm_dim*2, params.fc1_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(params.fc1_dim, params.outdim)
        
    def forward(self,bx):
        bs, seqlen, band_dim = bx.shape
        bx = bx.unsqueeze(1)
        # 
        x_pos_enc = self.make_posenc(bs, seqlen=seqlen, band_dim=band_dim)
        # concatenate the positional encodings with the mel
        bx_concat = torch.cat((x_pos_enc, bx),axis=1)
        o1 = self.convl11(bx_concat)
        o2 = self.convl12(o1) # 
        o3 = self.convl13(o2) # 
        o3 = o3.squeeze(-1)
        o = o3.transpose(-2,-1) # 
        _,(hs,_) = self.lstm(o) # 
        # concatenate both directions of LSTM
        s = torch.cat((hs[0],hs[1]),axis=1)
        # final LSTM
        out = self.fc2(self.relu(self.fc1(s)))
        return out
    
    def make_posenc(self, bs, seqlen, band_dim):
        posenc = make_pos_encodings(bs, seqlen, band_dim)
        return posenc.float()
```

Notice the positional encodings, a very simple X & Y vector added to all MEL-spectrograms automaticlaly within the forward pass of the DrumClassifier. See function `make_pos_encodings` in `code/pytorch_utils` for more details.

Default parameters are:

```python
params_dict = {'n_channels':3, # for MEL-spectrogram and X & Y positional-encodings
               "dropout_11": 0.05,
               "num_filters_11": 64,
               "kernel_size_11": (4,24),
               "stride_11": (1,2),
               "maxpool_size11": (1,2),
               "maxpool_stride11": (1,2),
               "dropout_12": 0.02,
               "num_filters_12": 96,
               "kernel_size_12": (3,7),
               "stride_12": (2,2),
               "maxpool_size12": (1,2),
               "maxpool_stride12": (1,2),
               "dropout_13": 0.05,
               "num_filters_13": 128,
               "kernel_size_13": (3,5),
               "stride_13": (2,1),
               "lstm_dim": 128,
               "fc1_dropout": 0.1,
               "fc1_dim": 128,
               "outdim": 30}
```

# What are the 30 percussion classes?
I used 30 instrument classes motivated by:
- availability in my personal collection of audio-files from many different sources.
- usefulness for industrial and cinematic music

I hope to expand these instruments in the future to include more tonal instruments. But in the meantime this is *CONDITIONAL* on the instrument being a percussive instrument.

```
'kick': kick drum
'tomf': floor tom
'thd': thud, deep distorted explosion
'tom': mid/low/high tom
'bass': tonal bass note,
'snr': snare drum
'tab': tabla
'taik': taiko drum
'cong': congo
'cah': industrial distorted snare/clap (think nine-inch nails snare)
'rim': rim-shot
'fx': other strange fx's
'wood': woodblocks
'ped': hihat peddle
'clav': clav,
'stick': snare-sticks or just sticks
'tim': timpani
'cui': cuica
'tri': triangle
'vib': vibraphone
'met': metal percussive sound
'hhc': hi-hat closed
'clp': clap,
'shk': shaker,
'gui': guiro or guira,
'whis': whistle,
'cow': cowbell,
'hho': high-hat open,
'rid': ride,
'cym': cymbal crash
```

# TRAINING (demo)
To see how the model was trained, have a look at the dummy-training file `training_demo.py` in `demo_sound_files`. For property-rights reasons, I cannot make the training data available.



TODO

# DEMO: AUTO-BUILD DRUMKITS

Some auxiliary functions are included that build drumkits in the Drumkv1 (I am also considering exportin to Hydrogen drumkits. The class `ClassiferPlusKv1Exporter` has functions to import samples, classify into instruments, and then assign instruments to specificy midi-keys. The arrangment is specified via a list that maps (midi)keys to instruments
python
```
MY_ASSIGNMENT = {'key_assignment': {'kick': 36, # assigns kick to midi-36
                                    'snr': 38, # snare to 38
                                    'hhc': 45, # high-hat closed to 45
				    ...
                                    'hho': 47}}
```

The functions `classify_and_make_kit` builds a single drumkit, while `classifies_and_make_kits` works in batch-mode, especially useful if you have lots of directories that each represent separate potential kits


```python
import sys
sys.path.append("code/")
from pytorch_utils import * # for the pytorch CNN-LSTM model
from drumclassifier_constants import * # constants
from drumclassifier_utils import * # make code
# load the drumkv1 classes and function
from drumkv1_constants import * 
from drumkv1_utils import * 

# make a configuration: maps midi keys to instruments (snr, kick, hihat, etc)
# (in the demo_sounds_files, we have a kick, hhc,hho and snare)
                           
MY_ASSIGNMENT = {'key_assignment': {'kick': 36, # assigns kick to midi-36
                                    'snr': 38, # snare to 38
                                    'hhc': 45, # high-hat closed to 45
                                    'hho': 47}} # high-hat open to

# in this demo, we only have the four instruments, but a fuller in arrangment can be seen in drumkv1_constants.DEFAULT_ASSIGNMENT

# intialize the XML-DRUMKV1 exporter
kit_maker = ClassiferPlusKv1Exporter(path_to_model = "models/mel_cnn_models/mel_cnn_model_high_v2.model", assignment_default = MY_ASSIGNMENT, verbose = 1)

# make a list of samplesinstruments for assignment
path_to_samples = ["demo_sound_files/kick/kick01.ogg","demo_sound_files/hhc/hihat_closed02.ogg","demo_sound_files/hho/hihat_opened01.ogg","demo_sound_files/snr/snare03.ogg"]

# classify and export to drumkv1 file
kit_maker.classify_and_make_kit(path_to_kit = "/tmp/my_first_drumkit.drumkv1", # exported kit
                                samples = path_to_samples) # list of samples

# if you want to auto-make many kits, using samples grouped into directories,
# ... then supply a list of directories to function 'classifies_and_makes_kits'
kit_maker.classifies_and_make_kits(path_to_kits = "export/path/to/kits/", # exported kit
                                directories = ["path/to/samples/", "path/to/samples2/"])

```

# Files

```
code/drumclassifier_utils.py	 # core classifier functionality 
code/drumclassifier_constants.py # classifier constants
code/pytorch_utils.py		 # pytorch mode syntax and classes
code/drumkv1_utils.py		 # (optional) Drumkv1 drumkit XML-exportation
code/drumkv1_constants.py	 # (optional) Drumkv1 drumkit constants
```