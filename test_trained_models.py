
import os
import sys
import json
from argparse import ArgumentParser, Namespace
import hashlib
import time
from time import gmtime, strftime
from IPython import embed
import logging

import numpy as np
import matplotlib.pyplot as plot
plot.switch_backend('agg')
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import cls_feature_class
import cls_data_generator
import parameters
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
from models import models
from criterions import MSELoss_ADPIT, SELLoss
from cls_dataset import * 
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
from torch_run import test_epoch



def init_logging_file(unique_hash_str):
    '''initiate the logging features, and the results will all dumped into results_audio/{unique_hash_str}'''
    os.makedirs(unique_hash_str, exist_ok=True)
    log_file = os.path.join(unique_hash_str, 'results_logs.log')
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def main(argv):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)
    model_folder = os.path.abspath(argv[2])
    results_folder = os.path.join(params['dcase_output_dir'], os.path.basename(model_folder)+f"_{params['model']}")
    train_splits, val_splits, test_splits = None, None, None
    log_file = os.path.join(results_folder, 'test_models_logs.log')
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

    if params['mode'] == 'dev':
        if '2020' in params['dataset_dir']:
            test_splits = [1]
            val_splits = [2]
            train_splits = [[3, 4, 5, 6]]

        elif '2021' in params['dataset_dir']:
            test_splits = [6]
            val_splits = [5]
            train_splits = [[1, 2, 3, 4]]

        elif '2022' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2023' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[1, 2, 3]] 
        elif '2024' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[3]]

        else:
            logging.info('ERROR: Unknown dataset splits')
            exit()
    if params['output_format'] == 'multi_accdoa':
        criterion = MSELoss_ADPIT()
    elif params['output_format'] == 'single_accdoa':
        criterion = nn.MSELoss()
    elif params['output_format'] == 'polar':
        criterion = SELLoss(params['unique_classes'])

    for split_cnt, split in enumerate(test_splits):
        logging.info('Loading validation dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
                params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
            )# init a class that generator the validation data split = [4] ,room4 
        # Collect i/o data size and load model configuration, load model weights to model
        # model we used are wrapped in models/ folder, you can modify it in parameter.py  
        if params['modality'] == 'audio_visual':
            data_in, vid_data_in, data_out = data_gen_test.get_data_sizes()
            model = models[params['model']](data_in, data_out, params, vid_data_in).to(device)
        else:
            data_in, data_out = data_gen_test.get_data_sizes() 
            model = models[params['model']](data_in, data_out, params).to(device)
        for model_name in os.listdir(model_folder):
            logging.info(f'model name {model_name}')
            model_name= os.path.join(model_folder, model_name)
            model.load_state_dict(torch.load(model_name, map_location='cpu'))
            
            logging.info('Loading unseen test dataset:')
            data_gen_test = cls_data_generator.DataGenerator(
                params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
            )
            test_dataset = SELDDataset(params, 'test')
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True) 
            score_obj = ComputeSELDResults(params)  #cls_compute_seld_results
            # Dump results in DCASE output format for calculating final scores  os.path.join(params['dcase_output_dir'], f'{unique_hash_str}',f'{unique_name}_test')
            dcase_output_test_folder = os.path.join(results_folder, f'test_model_{os.path.basename(model_name)}')
            cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
            logging.info('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

            # the results will be saved into *_test file
            test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
            # 2024 challenge only focus test_F, test_LE,  test_dist_err 

            use_jackknife=True
            test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )

            logging.info('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
            logging.info('SED metrics: F-score: {:0.1f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
            logging.info('DOA metrics: Angular error: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
            logging.info('Distance metrics: {:0.2f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
            logging.info('Relative Distance metrics: {:0.2f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.2f} , {:0.2f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

            if params['average']=='macro':
                logging.info('Classwise results on unseen test data')
                logging.info('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
                for cls_cnt in range(params['unique_classes']):
                    logging.info('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                        cls_cnt,

                        classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                    classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                    classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                    classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                        classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                    classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',

                        classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                        '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                    classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))

    
    logging.info("All the models in folders have been test.")


if __name__ == "__main__":
    
    # get_seld_result(sys.argv)
    try:
        import time 
        start = time.time()
        main(sys.argv)
        print( time.time()- start)
    except (ValueError, IOError) as e:
        sys.exit(e)
