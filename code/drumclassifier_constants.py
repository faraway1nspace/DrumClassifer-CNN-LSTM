PATH_TO_MODEL = "../models/mel_cnn_models/mel_cnn_model_high_v2.model"

PARAMS_DICT = {'n_channels':3,
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
               "outdim": 30}#len(set(ydata))} 

INSTRUMENT_NAMES = ['whis', 'met', 'cow', 'taik', 'clav', 'hhc', 'cah', 'tim', 'tab', 'kick', 'rid', 'bass', 'thd', 'hho', 'rim', 'shk', 'stick', 'tom', 'clp', 'wood', 'cong', 'snr', 'tri', 'vib', 'cym', 'gui', 'fx', 'tomf', 'ped', 'cui']


