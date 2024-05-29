# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
from datetime import datetime

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=True,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='3_1_dev_split0_multiaccdoa_foa_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='../Dataset/STARSS2023',  # server '../../dataset/STARSS2023' 

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='../Dataset/STARSS2023/feat_label_hnet/', # server '../../dataset/STARSS2023/feat_label_hnet/',

        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        filter = 'gammatone', # 'mel' / 'gammatone' / 'bark'
        fs=24000,  # sample rate
        hop_len_s=0.01, # ??TODO original 0.02 
        label_hop_len_s=0.1,  # resolution in annotation file  TODO origina
        max_audio_len_s=60, # length for each audio file
        nb_mel_bins=128,  # mel 64,128,256 TODO original 64 

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,


        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        
        # OUTPUT FORMAT 
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        output_format = 'single_accdoa', # 'single_accdoa', 'multi_accdoa'(adpit), polar

        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.
        
        seperate_polar = False, # Required for output_format = polar only, decide wether the azimuth and elevation should be pridicted seperatedly.



        # DNN MODEL PARAMETERS
        model = 'SeldModel',   # model will be trained, default: SeldModel, SeldConModel
        label_sequence_length=50,    # Feature sequence length 
        batch_size=128,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        nb_rnn_layers=2,
        rnn_size=128,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=250,  # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses event-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics

        # time used to generate hash value of results folder
        current_time = datetime.now().isoformat() 
    )

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '21':
        print("FOA + single ACCDOA\n + mel")
        params['quick_test'] = False
        params['filter'] = 'mel'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False 
        params['output_format'] = 'single_accdoa'
    
    elif argv == '32':
        print("FOA + single ACCDOA\n + gammatone")
        params['quick_test'] = False
        params['filter'] = 'gammatone'
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False 
        params['output_format'] = 'single_accdoa'

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['output_format'] = 'multi_accdoa'
        # params['finetune_mode'] = False
    elif argv == '31':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True
        params['output_format'] = 'multi_accdoa'
        params['finetune_mode'] = False
        params['models'] = 'SeldConModel'



    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['pretrained_model_weights'] = '6_1_dev_split0_multiaccdoa_mic_gcc_model.h5'
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    '''
    5 = 0.1 / 0.02 , 
    params['label_sequence_length'] = 50, first 
    params['label_hop_len_s'] is 100ms because the annotation file resulotion is 100ms, the relative attributes are:
        self._label_hop_len_s = params['label_hop_len_s']  # 0.1 second
        self._label_hop_len = int(self._fs * self._label_hop_len_s) # 2400  sample
        self._label_frame_res = self._fs / float(self._label_hop_len) # 10.0 , there are 10 label output in one second 
    feature_label_resolution: 
    params['hop_len_s'] is used to calculate the mel spectrum, and it is in second. The relative attributes are:
        self._hop_len = int(self._fs * self._hop_len_s) # 480
        self._win_len = 2 * self._hop_len # 960 
        self._nfft = self._next_greater_power_of_2(self._win_len) # 1024 
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    
    '''
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution # 50 * 5 
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling   [5, 1, 1]
    params['patience'] = int(params['nb_epochs'])  # Stop training if patience is reached 250
    params['model_dir'] = params['model_dir'] + '_' + params['modality']  # folder name of this 
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality'] # 

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2023' in params['dataset_dir']:
        params['unique_classes'] = 13
    elif '2024' in params['dataset_dir']:
        params['unique_classes'] = 13
    
    # print params 
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
