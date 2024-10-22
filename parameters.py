# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.
from datetime import datetime

# classes according to the audioset
sound_classes = {
    'speech' : 0, # speech, laughter, whisper, crying, singing
    'human_voice': 1, # clearthroat, cough
    'hands': 2,# Finger snapping, Clapping
    'walk_footsteps' : 3,# walk,footsteps
    'telephone':4,  # phone, telephone
    'doors' :5, # doorslam, open close
    'cupboard_drawer': 6,# 
    'water_faucet':7, 
    'domestic_sounds': 8, #Scissors, writing , computer keyboard, pageturn, printer
    'bells': 9,
    'keys': 10, #keys drop 
    'glass': 11, #chink_chinl
    'konck': 12,
    'music':13, 
    'music_instrument': 14 
}

STARSS_sound_classes_mapping = {
    0: 0,
    1: 0,
    2: 2,
    3: 4,
    4: 0,
    5: 8,
    6: 3,
    7: 5, 
    8: 13,
    9: 14, 
    10: 7,
    11: 9,
    12 : 12
}

ANSYN_ASIRD_sound_classes_mapping = {
    'keyboard': 8, 
    'doorslam': 5, 
    'laughter': 0, 
    'clearthroat': 1, 
    'pageturn': 8, 
    'speech': 0, 
    'phone': 4, 
    'cough': 1, 
    'drawer': 6, 
    'keysDrop': 10, 
    'knock': 12
}

L3DAS21_sound_class_mapping = {
    'Writing': 8, 
    'Computer_keyboard': 8, 
    'Cupboard_open_or_close': 6, 
    'Female_speech_and_woman_speaking': 0, 
    'Printer': 8, 
    'Chink_and_clink': 11, 
    'Scissors': 8, 
    'Knock': 12, 
    'Keys_jangling': 10, 
    'Finger_snapping': 2, 
    'Telephone': 4, 
    'Drawer_open_or_close': 6, 
    'Male_speech_and_man_speaking': 0, 
    'Laughter': 0
}
 

def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    
    params = dict( 
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs
        finetune_mode=False,  # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='./models_audio/3_1_dev_split0_multi_accdoa_STARSS2023_SeldModel.h5',
        datasets_dir_dic = {'ANSYN': '../Dataset/ANSYN', 'ASIRD': '../Dataset/ASIRD', 'L3DAS21': '../Dataset/L3DAS21', 'STARSS2023': '../../dataset/STARSS2023'},
        feat_label_dir_dic = {'ANSYN': '../Dataset/ANSYN/feat_label_hnet', 'ASIRD': '../Dataset/ASIRD/feat_label_hnet', 'L3DAS21': '../Dataset/L3DAS21/feat_label_hnet', 'STARSS2023': '../../dataset/STARSS2023/feat_label_hnet'},
        fs_dic = {'ANSYN': 44100, 'ASIRD': 44100, 'L3DAS21': 32000, 'STARSS2023':24000},
        max_audio_len_s_dic = {'ANSYN': 30 , 'ASIRD': 30, 'L3DAS21': 60, 'STARSS2023':60},
        subset = False,
        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders,
        
    
        
        model_dir='models',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        synthetic = False, 
        data_type ='foa',  # 'foa' - ambisonic or 'mic' - microphone signals
        dataset = 'ANSYN', # ANSYN, ASIRD, L3DAS21, STARSS2023, and this need to be a string 

        # FEATURE PARAMS
        preprocessing_type = 'iv_7', # Currently, this system only support iv_7.
        data_augmentation = False, # if this variarble is set to be true, then the function 'dev_feat_cls.extract_all_feature()' will generate all these features
        sound_scaper = False,
        filter = 'gammatone', # Alternative filter types are: 'mel' / 'gammatone' / 'bark'
        hop_len_s=0.005, # ??
        label_hop_len_s=0.05,  # resolution in annotation file
        label_hop_len_s_STARSS  = 0.1, # resolution in starss annotation file, only need by starss 
        max_audio_len_s = 60, # length for each audio file
        nb_mel_bins=128,  # mel,128

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # CSTmodel 
        baseline = True,
        encoder = 'conv',           # ['conv', 'ResNet', 'SENet']
        LinearLayer = False,        # Linear Layer right after attention layers (usually not used/employed in baseline model)
        FreqAtten = False,          # Use of Divided Spectro-Temporal Attention (DST Attention)
        ChAtten_DCA = False,        # Use of Divided Channel-S-T Attention (CST Attention)
        ChAtten_ULE = False,        # Use of Divided C-S-T attention with Unfold (Unfolded CST attention)
        CMT_block = False,          # Use of LPU & IRFNN
        CMT_split = False,          # Apply LPU & IRFNN on S, T attention layers independently
        t_pooling_loc = 'front',                           # False - Single-ACCDOA or True - Multi-ACCDOA            # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.
        self_attn=True,



        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        # MODEL TYPE
        modality='audio',  # 'audio' or 'audio_visual'
        
        # OUTPUT FORMAT 
        per_file = True,
        multi_accdoa=True,  # False - Single-ACCDOA or True - Multi-ACCDOA
        output_format = 'multi_accdoa', # 'single_accdoa', 'multi_accdoa'(adpit), polar

        thresh_unify=15,    # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.
        
        seperate_polar = False, # Required for output_format = polar only, decide wether the azimuth and elevation should be pridicted seperatedly.

        # conformer layer
        cf_num_heads = 4,      #TODO
        cf_ffn_dim = 512,
        cf_num_layers = 2,
        cf_depthwise_conv_kernel_size = 31,
        cf_num_cfs = 2,

        # DNN MODEL PARAMETERS
        model = 'SeldModel',   # model will be trained, default: SeldModel, SeldConModel
        label_sequence_length=50,    # Feature sequence length 
        batch_size=256,              # Batch size
        dropout_rate=0.05,           # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,           # Number of CNN nodes, constant for each layer TODO
        f_pool_size=[4, 4, 2],      # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        scconv = False,
        mhsa = False, 
        residual_connections = False,
        nb_heads=8,
        nb_self_attn_layers=2,
        nb_transformer_layers=2,

        positoncal_encoding = False,
        learnable_pe = False, 
        nb_rnn_layers=2,
        rnn_size=256,

        nb_fnn_layers=1,
    
        # HYPER PARAMETERS
        nb_epochs=1000,  # Train for maximum epochs
        nb_early_stop_patience = 50,  # if the validation loss have not improved for 50 epochs, then stop trainning 
        write_output_file_patience = 5,
        lr=1e-3,

        # METRIC
        average='macro',                 # Supports 'micro': sample-wise average and 'macro': class-wise average,
        segment_based_metrics=False,     # If True, uses segment-based metrics, else uses event-based metrics
        evaluate_distance=True,          # If True, computes distance errors and apply distance threshold to the detections
        lad_doa_thresh=20,               # DOA error threshold for computing the detection metrics
        lad_dist_thresh=float('inf'),    # Absolute distance error threshold for computing the detection metrics
        lad_reldist_thresh=float('1'),  # Relative distance error threshold for computing the detection metrics

        # time used to generate hash value of results folder
    )   
    
    # ########### User defined parameters ##############
    params['feature_label_resolution'] = int(params['label_hop_len_s'] // params['hop_len_s']) 
    if argv == '1':
        print("Generate intermediate audio file\n")  #SELD2024bark300:99
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 2048
        # params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'VanillaSeldModel'
        # params['subset'] =  True
        params['rnn_size'] =128
    
    elif argv == '110':
        print("Generate intermediate audio file\n")  #SELD2024, bark300:99
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =100 # 20
        params['batch_size'] = 256 
        # params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'VanillaSeldModel'
        # params['subset'] =  True
        params['rnn_size'] =128

    elif argv == '111':
        print("Generate intermediate audio file\n")  #SELD2024, bark300:99
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =40 # 20
        params['batch_size'] = 256 
        # params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'VanillaSeldModel'
        # params['subset'] =  True
        params['rnn_size'] =128

    elif argv == '12':
        print("Generate intermediate audio file\n")  #GSELD2024 TCN
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'GSELD'
        params['subset'] =  True
        params['rnn_size'] =128

        params['tcn_channels'] = [20, 64, 128, 64, 20]
        params['tcn_kernel_size'] = 3
    
    elif argv == '121':
        print("Generate intermediate audio file\n")  #GSELD2024 TCN 20, 32, 64, 32, 20]
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'GSELD'
        params['subset'] =  True
        params['rnn_size'] =128

        params['tcn_channels'] = [20, 32, 64, 32, 20]
        params['tcn_kernel_size'] = 3

    elif argv == '122':
        print("Generate intermediate audio file\n")  #GSELD2024 TCN 20, 32, 64, 32, 20] 4
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['filter'] = 'bark'
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2
        params['per_file'] = True
        params['model'] = 'GSELD'
        params['subset'] =  True
        params['rnn_size'] =128

        params['tcn_channels'] = [20, 32, 64, 32, 20]
        params['tcn_kernel_size'] = 4

    elif argv == '13':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['per_file'] = True
        # params['subset'] =  True

        params['model'] = 'CST_former'
        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True
        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

    elif argv == '14':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['per_file'] = True
        # params['subset'] =  True
        params['filter'] = 'gammatone'

        params['model'] = 'CST_former'
        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True
        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

        params['conv_type'] = 'scconv' # 'scconv' 'srucru' 'origin'


    elif argv == '15':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['per_file'] = True
        # params['subset'] =  True
        params['filter'] = 'gammatone'

        params['model'] = 'CST_former'
        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True
        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

        params['conv_type'] = 'origin' # 'scconv' 'srucru' 'origin'
    elif argv == '16':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['per_file'] = True
        # params['subset'] =  True
        params['filter'] = 'bark'

        params['model'] = 'CST_former'
        params['FreqAtten'] = True
        params['ChAtten_ULE'] = True
        params['CMT_block'] = True
        params["f_pool_size"] = [1,2,2] 
        params['t_pool_size'] = [1,1, params['feature_label_resolution']]

        params['conv_type'] = 'origin' # 'scconv' 'srucru' 'origin'


        
    elif argv == '2':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 256
        params['lr'] /= 4
        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformerEdit'

        params['use_efficientnet'] = True
        params['use_se_block'] = False



    elif argv == '21':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  False
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['feature_extraction'] = 'conv'
        params['use_efficientnet'] = False
        params['use_se_block'] = False

    
    elif argv == '211':
        print("Seld conformer with weights initiate\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['init_weights'] = True
        params['use_efficientnet'] = False
        params['use_se_block'] = False

    elif argv == '2111':
        print("Generate intermediate audio file\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2
        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 256 #128




    elif argv == '3': # not per file, batch size, 128 bins, 1s 
        print("Seld conformer with weights initiate\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['init_weights'] = True
        params['use_efficientnet'] = False
        params['use_se_block'] = False

    elif argv == '31': # new conformer
        print("Seld conformer with weights initiate\n")
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #256

        params['init_weights'] = False
        params['use_efficientnet'] = False
        params['use_se_block'] = False

    elif argv == '32':
        print("Generate intermediate audio file\n")  # 8 layer conformer
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128
        params['cf_num_cfs'] = 8

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False


    elif argv == '33':
        print("Generate intermediate audio file\n")  # 4 layer conformer 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128
        params['cf_num_cfs'] = 4

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
    
    elif argv == '34':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '35':
        print("Generate intermediate audio file\n")  # resnet 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['feature_extraction'] = 'resnetf'
        params['use_se_block'] = False

    elif argv == '36':
        print("Generate intermediate audio file\n")  # resnet 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = True

    elif argv == '37':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = True

    elif argv == '38':
        params['filter'] = 'bark'  #TODO
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        # params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False


    elif argv == '39':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = True

    elif argv == '310':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128
        params['cf_ffn_dim'] = 1024 # 512

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = True

    elif argv == '311':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128
        params['cf_num_cfs'] = 1

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = True

    elif argv == '312': # pe
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        params['positoncal_encoding'] = True
        params['learnable_pe'] = False 
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
    elif argv == '313':  # learnable pe
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 1024
        params['positoncal_encoding'] = True
        params['learnable_pe'] = True
        # params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 64 # 64
        params['rnn_size'] = 128 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '314':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
        params['cf_num_cfs'] = 1 

    elif argv == '315':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['cf_num_layers'] = 4 
        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False


    elif argv == '316':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['cf_num_heads'] = 8
        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '317':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
        params['residual_connections'] = True
        

    elif argv == '318':
        print("Generate intermediate audio file\n")  # rnn size 256 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
        params['mhsa'] = True

    elif argv == '319':
        print("Generate intermediate audio file\n")  # use scconv block 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False
        params['scconv'] = True

    elif argv == '320':
        print("Generate intermediate audio file\n")  # use scconv block 
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        # params['subset'] =  True
        params['model'] = 'SELDConformer'
        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '321':
        print("Generate intermediate audio file\n")  # GSELD
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'GSELD'


        params['tcn_channels'] = [20, 64, 128, 64, 20]
        params['tcn_kernel_size'] = 4

        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '322':
        print("Generate intermediate audio file\n")  #GSELD kernel size 3
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'GSELD'


        params['tcn_channels'] = [20, 64, 128, 64, 20]
        params['tcn_kernel_size'] = 3

        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False


    elif argv == '323':
        print("Generate intermediate audio file\n")  #GSELD  [20, 32, 64, 32, 20] [20, 40, 80, 40, 20]
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'GSELD'


        params['tcn_channels'] = [20, 32, 64, 32, 20]
        params['tcn_kernel_size'] = 3

        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

    elif argv == '324':
        print("Generate intermediate audio file\n")  #GSELD  [20, 40, 80, 40, 20]
        params['quick_test'] = False
        params['data_type'] = 'foa'
        params['multi_accdoa'] = True
        params['data_augmentation'] = False 
        params['synthetic'] = True 
        params['output_format'] = 'multi_accdoa' 
        params['hop_len_s'] = 0.01   
        params['label_hop_len_s'] =0.05  # resolution in annotation file
        params['dataset'] = 'STARSS2023'
        params['label_sequence_length'] =20
        params['batch_size'] = 512
        params['lr'] /= 2

        params['per_file'] = True
        params['subset'] =  True
        params['model'] = 'GSELD'


        params['tcn_channels'] = [20, 40, 80, 40, 20]
        params['tcn_kernel_size'] = 3

        params['nb_cnn2d_filt'] = 128 # 64
        params['rnn_size'] = 256 #128

        params['feature_extraction'] = 'conv'
        params['use_se_block'] = False

        
    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])  
    '''
    data_in: (570, 7, 100, 128) feature_label_resolution is 0.05 / 0.01 = 5, 
    data_out: (570, 20, 180) 
    the feature_sequence_length is 100 = params['label_sequence_length'] * feature_label_resolution = 20 * 5 = 100 
    '''
    '''f
    5 = 0.1 / 0.02 , 
    params['label_sequence_length'] = 50, first, output timestep is 50 that is 50 * 100 = 5000ms = 5s  ?
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
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution # original:50 * 5, new:    20 * 5  
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling   [5, 1, 1]
    params['model_dir'] = params['model_dir'] + '_' + params['modality']  # folder name of this 
    params['dcase_output_dir'] = params['dcase_output_dir'] + '_' + params['modality'] # 
    params['segment_length'] = params['label_sequence_length'] * params['label_hop_len_s']

    # if '2020' in params['dataset_dir']:
    #     params['unique_classes'] = 14
    # elif '2021' in params['dataset_dir']:
    #     params['unique_classes'] = 12
    # elif '2022' in params['dataset_dir']:
    #     params['unique_classes'] = 13
    # elif '2023' in params['dataset_dir']:
    #     params['unique_classes'] = 13
    # elif '2024' in params['dataset_dir']:
    #     params['unique_classes'] = 13
    # else: # ANSYN, ASIRD, L3DAS21, 
    #     params['unique_classes'] = 13 
    params['dataset_dir'] =  params['datasets_dir_dic'][params['dataset']]  # server '../../dataset/STARSS2023' ' 
    params['feat_label_dir'] =  params['feat_label_dir_dic'][params['dataset']] # server '../../dataset/STARSS2023/feat_label_hnet/',
    params['fs'] = params['fs_dic'][params['dataset']]
    params['max_audio_len_s'] = params['max_audio_len_s_dic'][params['dataset']]

    if params['dataset'] in ['ANSYN', 'ASIRD']:
        params['classes_mapping'] = ANSYN_ASIRD_sound_classes_mapping
    elif params['dataset'] in ['L3DAS21']:
        params['classes_mapping'] = L3DAS21_sound_class_mapping
    elif params['dataset'] in ['STARSS2023']:
        params['classes_mapping'] = STARSS_sound_classes_mapping
    params['unique_classes'] = 13
    
    # print params 
    print('-------------------info of parameters-------------------')
    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    print('---------------end of info parameters-------------------')
    return params





if __name__ == '__main__':
    params = get_params('3')
    print(params)
