# DrumClassifier with an CNN-LSTM (pytorch)

This is a deep-learning solution to help organizing my drum-sample collection (audio-files like kicks, snares, cymbals, claps, high-hats, etc.). Searching for the perfect-sounding snare during music production really kills my creativity, so I build this application (and associated tools) to auto-build some custom drumkits.

If you are interested in using this in your work/project, I am open to collaboration and development. My interests:
 - integrate into a DAW as a better-file-browser (like a VST or LV2 plugin)
 - visualize sampels according to their CNN-LSTM latent representations (like a NN-PCA)

### Purpose:
- classify drum-samples (.wav, flac, ogg, aiff) into 30 percussion classes
- (optional) auto-build drumkits based on predicted classes and export to DrumKV1 kit (a lv2 plugin)
- (future) explore audio-samples according to latent dimensions

### Method:
- python/pytorch based
- a CNN-LSTM 3-layer neural network classifier
- uses MEL-frequency coefficients as the input

### Dependencies
- librosa
- pytorch
- numpy

# Example usage
```foobar = dooo```


# Model

The core of the classifier is CNN-LSTM.
- CNN: a stack of three filters, running over the MEL-spectrum and time-dimension. 
- LSTM: collects the 1-D outputs of the CNN and runs them through a final RNN. This allows for time-varying MEL-spectrograms (rather than for artificially fixed-time-length audio files).

### Model in pytorch:
see the file `models/mel_cnn_models/mel_cnn_model_high_v2.model`

```
class Classifier(nn.Module):   
    def __init__(self,params):
        super(Classifier, self).__init__()
        
        # convolution
        self.convl11 = nn.Sequential(nn.Dropout(params.dropout_11),
                        nn.Conv2d(params.n_channels, params.num_filters_11, kernel_size=params.kernel_size_11, stride=params.stride_11),
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

Notice the positional encodings, a very simple X & Y dimension added to all MEL-spectrograms automaticlaly within the forward pass of the DrumClassifier. See function `make_pos_encodings` in `code/pytorch_utils` for more details.

Default parameters are:

```
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

