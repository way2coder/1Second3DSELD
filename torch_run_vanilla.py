#
# A wrapper script that trains model. The training stops when the early stopping metric - SELD error stops improving.
#

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


def get_experiment_hash(params):
    """Generates a unique hash-value depending on the provided experimental parameters."""
    return hashlib.md5(
        json.dumps(params, sort_keys=True).encode("utf-8")
    ).hexdigest()


def get_accdoa_labels(accdoa_in, nb_classes, output_format):
    # accdoa_in.shape (batchsize, T/5, 52)
    if output_format == 'single_accoda':
        x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
        sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

        
    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1<0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes: 7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:] 
    dist2[dist2<0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]
    # multi tracks, sed(sound event detection) b, 50, 13 doa b, 50, 39 dist b, 50, 13    bool float32 float32
    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0

def polar_to_cartesian(azi,ele):
    x = np.sin(ele) * np.cos(azi)
    y = np.sin(ele) * np.sin(azi)
    z = np.cos(ele)
    return x, y, z



def train_epoch(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for values in data_generator.generate():
        # load one batch of data
        if len(values) == 2:
            data, target = values
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
            optimizer.zero_grad()
            output = model(data)
        elif len(values) == 3:
            data, vid_feat, target = values
            data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
            optimizer.zero_grad()
            output = model(data, vid_feat)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()
    # inference_time, loss_time, reshape_time, write_time = 0., 0., 0., 0.
    import time 

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for values in data_generator.generate(): # for ebatch
            if len(values) == 2:
                data, target = values    # data[(114, 7, 250, 64)] target[(114, 50, 6, 5, 13)]
                data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
                # data, target = data.to(device), target.to(device)
                output = model(data) # multiaccdoa output ([114, 50, 156]) single accodoa batchsize, T/5, 65
            elif len(values) == 3:
                data, vid_feat, target = values
                data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()

                output = model(data, vid_feat) 
            # inference_time += time.time() - start_time

            start_time = time.time()
            loss = criterion(output, target)
            # loss_time += time.time() - start_time

            if params['output_format'] == 'multi_accdoa':
                sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0) # 5700, 13
                doa_pred0 = reshape_3Dto2D(doa_pred0) # 5700, 39
                dist_pred0 = reshape_3Dto2D(dist_pred0) # 5700, 13
                sed_pred1 = reshape_3Dto2D(sed_pred1) 
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                dist_pred1 = reshape_3Dto2D(dist_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
                dist_pred2 = reshape_3Dto2D(dist_pred2)
            elif params['output_format'] == 'single_accdoa': #output (b, 50, 13*4)
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'], params['output_format'])
                sed_pred = reshape_3Dto2D(sed_pred) # time, 13
                doa_pred = reshape_3Dto2D(doa_pred)  # time 
            elif params['output_format'] == 'polar': #output b, 50, 13, 4
                sed_pred , doa_pred, dist_pred = output.detach().cpu().numpy()[...,0], output.detach().cpu().numpy()[...,1:3], output.detach().cpu().numpy()[...,-1]
                sed_pred = reshape_3Dto2D(sed_pred) # time, 13
                doa_pred = reshape_3Dto2D(doa_pred)  # time 
                dist_pred = reshape_3Dto2D(dist_pred) 
            
            # reshape_time += time.time() - start_time
            # dump SELD results to the correspondin file
            # file nameformat 3_1_dev_split0_multiaccdoa_foa_val\\fold4_room10_mix003.csv   test_filelist[file_cnt] --- 'fold4_room10_mix003.npy'
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv')) 
            file_cnt += 1 # file_cnt update
            output_dict = {} # initiate per batch 
            if params['output_format'] == 'multi_accdoa':
                for frame_cnt in range(sed_pred0.shape[0]): # for each batchsize & timestep 
                    for class_cnt in range(sed_pred0.shape[1]): # for each class 
                        # determine whether track0 is similar to track1 
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:  # each track is not similar with the other track 
                            if sed_pred0[frame_cnt][class_cnt]>0.5: 
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred2[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred0[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']], dist_pred1[frame_cnt][class_cnt]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                dist_pred_fc = (dist_pred2[frame_cnt] + dist_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            dist_pred_fc = (dist_pred0[frame_cnt] + dist_pred1[frame_cnt] + dist_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']], dist_pred_fc[class_cnt]])
            elif params['output_format'] == 'single_accdoa':
                for frame_cnt in range(sed_pred.shape[0]): 
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]]) 
                            # 
            elif params['output_format'] == 'polar':
                for frame_cnt in range(sed_pred.shape[0]): 
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            x, y, z = polar_to_cartesian(doa_pred[frame_cnt][class_cnt][0],doa_pred[frame_cnt][class_cnt][1])
                            output_dict[frame_cnt].append([class_cnt, x, y, z, dist_pred[frame_cnt][class_cnt]]) 
                            # 
            # start_time = time.time()
            data_generator.write_output_format_file(output_file, output_dict, params['output_format'])


            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break
        test_loss /= nb_test_batches
    # print(f'Time for each stage are{inference_time},{loss_time},{reshape_time},{write_time}') Time for each stage are3.2587361335754395,0.1810741424560547,5.755268812179565,51.642672300338745
    return test_loss


def train_epoch(train_loader, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()  # set to train model 
    
    for batch_idx, values in enumerate(train_loader):
        # load one batch of data
        if len(values) == 2: 
            # breakpoint()
            data, target = values   # data always be (batchsize, 7, 250, 64)  target: single_accdoa(batchsize, 50, 52) multi_accdoa (batchsize, 50, 6, 5, 13)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # breakpoint()
            output = model(data)  # TODO:model should be modify 
        elif len(values) == 3:
            data, vid_feat, target = values
            # data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
            optimizer.zero_grad()
            output = model(data, vid_feat)

        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break
    # final loss equals accmulative train_loss divid nb_tarin_batches
    train_loss /= nb_train_batches

    return train_loss


def validation_epoch(train_loader, optimizer, model, criterion, params, device     ):
    nb_train_batches, train_loss = 0, 0.
    model.eval()  # set to evaluation model 
    with torch.no_grad(): 
        for batch_idx, values in enumerate(train_loader):
            # load one batch of data
            if len(values) == 2: 
                # breakpoint()
                data, target = values   # data always be (batchsize, 7, 250, 64)  target: single_accdoa(batchsize, 50, 52) multi_accdoa (batchsize, 50, 6, 5, 13)
                # data, target = torch.tensor(data).clone().detach().to(device).float(), torch.tensor(target).clone().detach().to(device).float()
                data, target = data.to(device), target.to(device)
                # optimizer.zero_grad()
                # breakpoint()
                output = model(data)  # TODO:model should be modify 
            elif len(values) == 3:
                data, vid_feat, target = values
                # data, vid_feat, target = torch.tensor(data).to(device).float(), torch.tensor(vid_feat).to(device).float(), torch.tensor(target).to(device).float()
                # optimizer.zero_grad()
                output = model(data, vid_feat)

            
            loss = criterion(output, target)
            # loss.backward()
            # optimizer.step()
            
            train_loss += loss.item()
            nb_train_batches += 1
            if params['quick_test'] and nb_train_batches == 4:
                break
        # final loss equals accmulative train_loss divid nb_tarin_batches
        train_loss /= nb_train_batches

    return train_loss



def should_save_model(val_F, val_LE, val_rel_dist_err, best_F, best_LE, best_rel_dist_err):

    if val_F > best_F or ((best_F - val_F) < 0.01 and val_LE < best_LE and val_rel_dist_err < best_rel_dist_err):
        return True
    else: 
        return False

def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # generate unique_hash_str according to params and initiate logging files 
    unique_hash_str = get_experiment_hash(params=params)
    '''initiate the logging features, and the results will all dumped into results_audio/{unique_hash_str}'''
    results_folder = os.path.join(params['dcase_output_dir'], f"{unique_hash_str}_{params['model']}")
    os.makedirs(results_folder, exist_ok=True)

    log_file = os.path.join(results_folder, 'results_logs.log')
    # if not os.path.exists(log_file):
    #     open(log_file, 'a').close()  # Create the file if it does not exist
    #     os.chmod(log_file, 0o644)  # Set permissions if necessary
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='a', 
                    format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

    logging.info(argv)
    if len(argv) != 3:
        logging.info('\n\n')
        logging.info('-------------------------------------------------------------------------------------------------------')
        logging.info('The code expected two optional inputs')
        logging.info('\t>> python seld.py <task-id> <job-id>')
        logging.info('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        logging.info('Using default inputs for now')
        logging.info('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        logging.info('-------------------------------------------------------------------------------------------------------')
        logging.info('\n\n')
    for key, value in params.items():
        logging.info("\t{}: {}".format(key, value))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # torch.autograd.set_detect_anomaly(True) 
    torch.autograd.set_detect_anomaly(True)

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
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
            test_splits = [[2]]
            val_splits = [[2]]
            train_splits = [[1, 3]] 
        elif '2024' in params['dataset_dir']:
            test_splits = [[4]]
            val_splits = [[4]]
            train_splits = [[3]]

        else:
            logging.info('ERROR: Unknown dataset splits')
            exit()
    # train a seperate model for each test_splits
    for split_cnt, split in enumerate(test_splits):
        logging.info('\n\n---------------------------------------------------------------------------------------------------')
        logging.info('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        logging.info('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        # loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, params['output_format'], loc_feat
        )
        models_dir = os.path.join(params['model_dir'], unique_hash_str)
        cls_feature_class.create_folder(models_dir)

        best_models = []
        max_models = 3  # 保存模型的最大数量 Max number of models saved in the list

        logging.info(f"all the models will be saved in : {models_dir}")
                     
        model_name = '{}_{}.h5'.format(os.path.join(params['model_dir'], unique_name),params['model'])
        logging.info("unique_name: {}\n".format(unique_name))

        # Load train and validation data,
        logging.info('Loading training dataset:')
        
        train_dataset = SELDDataset(params, 'train')
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, num_workers= 16) 

        # validation_dataset = SELDDataset(params, 'val')
        # validation_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=16) 



        # data_gen_train = cls_data_generator.DataGenerator(
        #     params=params, split=train_splits[split_cnt]
        # ) # init a class that generator the train data, split = [1, 2, 3]

        logging.info('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=params['per_file'], mode = 'test'
        ) # init a class that generator the validation data split = [4] ,room4 

        # Collect i/o data size and load model configuration, load model weights to model
        # model we used are wrapped in models/ folder, you can modify it in parameter.py  
        if params['modality'] == 'audio_visual':
            data_in, vid_data_in, data_out = data_gen_val.get_data_sizes()  
            model = models[params['model']](data_in, data_out, params, vid_data_in).to(device)
        else:
            data_in, data_out = data_gen_val.get_data_sizes() 
            # data_in  tuple (570, 7, 100, 128)
            # data_out tuple (20, 180) 
            model = models[params['model']](data_in, data_out, params).to(device)
        
        if params['finetune_mode']:
            logging.info('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights']))
            state_dict = torch.load(params['pretrained_model_weights'], map_location='cpu') 
            if params['modality'] == 'audio_visual':
                state_dict = {k: v for k, v in state_dict.items() if 'fnn' not in k}
            model.load_state_dict(state_dict, strict=False)

        logging.info('---------------- SELD-net -------------------')
        logging.info('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        logging.info('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n, rnn_size: {}\n, nb_attention_blocks: {}\n, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'], params['nb_self_attn_layers'],
            params['fnn_size']))
        logging.info(model)
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f'total parameters of the model:{total_params}')
        


        # Dump results in DCASE output format for calculating final scores
        
    
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], f"{unique_hash_str}_{params['model']}",f'{unique_name}_val')
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        logging.info('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class, use a class to evaluate the params
        score_obj = ComputeSELDResults(params, train_dataset._feat_cls._new_label_dir)  #cls_compute_seld_results

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err, best_rel_dist_err = 1., 0., 180., 0., 9999, 999999., 999999.
        best_val_loss = 999999.
        patience_cnt = 0
        metrics_history = np.empty((0, 3))  # save the main 3 metrics required by this year's challenge

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=params['nb_epochs'])
        # criterion
        if params['output_format'] == 'multi_accdoa':
            criterion = MSELoss_ADPIT()
        elif params['output_format'] == 'single_accdoa':
            criterion = nn.MSELoss()
        elif params['output_format'] == 'polar':
            criterion = SELLoss(params['unique_classes'])
        lr = []
        for epoch_cnt in range(nb_epoch):
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(train_loader, optimizer, model, criterion, params, device) # only the train_loss is needed in training procedure
            train_time = time.time() - start_time
            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------

            start_time = time.time()
            val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            # val_loss = test_epoch(data_gen_val, optimizer, model, criterion, params, device)   #(train_loader, optimizer, model, criterion, params, device):
            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            val_time = time.time() - start_time
            
            start_time = time.time()
            val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

            metric_time = time.time() - start_time

            schedular.step()
            # best_val_loss = val_loss if val_loss < best_val_loss else best_val_loss
            best_F = val_F if val_F > best_F else best_F
            # metrics_history = update_metrics_history(metrics_history, val_F, val_LE, val_rel_dist_err)
            # F1score: val_F DOA angular error:val_LE relative distance error: val_rel_dist_err
            
            if epoch_cnt > 10:
                if (len(best_models) < max_models or val_F > best_models[-1][0]) :
                    best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr, best_dist_err = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr, val_dist_err
                    # Save the new model
                    model_path = os.path.join(models_dir, str(epoch_cnt) + '.h5')
                    # model_name.replace('.h5', f'_epoch{epoch_cnt}_loss{val_loss:.4f}.h5')
                    torch.save(model.state_dict(), model_path)
                    #Update the model list
                    if len(best_models) < max_models:
                        best_models.append((val_F, model_path))
                    else:
                        # Remove the worst model
                        os.remove(best_models[0][1])
                        best_models[0] = (val_F, model_path)
                    
                    # Sort the model list
                    best_models.sort(key=lambda x: x[0])
                    patience_cnt = 0
                else:
                    patience_cnt += 1

            # logging.info stats
            # logging.info(
            #     'epoch: {}, time: {:0.3f}/{:0.2f}/{:0.2f}, '
            #     'train_loss: {:0.4f}, val_loss: {:0.4f}, '
            #     'F/AE/Dist_err/Rel_dist_err/SELD: {}, '
            #     'best_val_epoch: {} {}'.format(
            #         epoch_cnt, train_time, val_time,metric__time,
            #         train_loss, val_loss,
            #         '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr),
            #         best_val_epoch,
            #         '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format( best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr))
            # )
            logging.info(
                f'epoch: {epoch_cnt}, train_time/val_time/metric_time: {train_time:0.2f}/{ val_time:0.2f}/{metric_time:0.2f}, '
                f'train_loss: {train_loss:0.4f}, val_loss: {val_loss:0.4f}, '
                # 'F/AE/Dist_err/Rel_dist_err/SELD: {}, '
                # 'best_val_epoch: {} {}'.format(
                #     epoch_cnt, train_time, val_time,metric_time,
                #     train_loss, val_loss,
                #     '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr),
                #     best_val_epoch,
                #     '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.3f})'.format( best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr))
                f'current_metircs: val_F, val_LE, val_dist_err, val_rel_dist_err, val_seld_scr{val_F:0.4f}/{val_LE:0.3f}/{val_dist_err:0.3f}/{val_rel_dist_err:0.3f}/{val_seld_scr:0.3f} '
                f'best_metrics:best_F, best_LE, best_dist_err, best_rel_dist_err, best_seld_scr{best_F:0.4f}/{best_LE:0.3f}/{best_dist_err:0.3f}/{best_rel_dist_err:0.3f}/{best_seld_scr:0.3f}'
            )

            if patience_cnt > params['nb_early_stop_patience']:
                break

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        logging.info('Load best model weights')
        # for model_name in  best_models:
        model_name = best_models[-1][1]
        logging.info(f'model name {model_name}')
        model_name = best_models[-1][1]
        model.load_state_dict(torch.load(model_name, map_location='cpu'))  # TODO:Move the model to GPU to accelerate the inference speed.
        model.to(device)

        logging.info('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=params['per_file'] ,mode = 'test'
        )
        # test_dataset = SELDDataset(params, 'test')
        # test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=True) 
        
        # Dump results in DCASE output format for calculating final scores  os.path.join(params['dcase_output_dir'], f'{unique_hash_str}',f'{unique_name}_test')
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'], f"{unique_hash_str}_{params['model']}",f'{unique_name}_test')
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        logging.info('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        # the results will be saved into *_test file
        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)
        # 2024 challenge only focus test_F, test_LE,  test_dist_err 

        use_jackknife=True
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )

        logging.info('SELD score (early stopping metric): {:0.3f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.3f}, {:0.3f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        logging.info('SED metrics: F-score: {:0.1f} {}'.format(100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.3f}, {:0.3f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        logging.info('DOA metrics: Angular error: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.3f} , {:0.3f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else ''))
        logging.info('Distance metrics: {:0.3f} {}'.format(test_dist_err[0] if use_jackknife else test_dist_err, '[{:0.3f} , {:0.3f}]'.format(test_dist_err[1][0], test_dist_err[1][1]) if use_jackknife else ''))
        logging.info('Relative Distance metrics: {:0.3f} {}'.format(test_rel_dist_err[0] if use_jackknife else test_rel_dist_err, '[{:0.3f} , {:0.3f}]'.format(test_rel_dist_err[1][0], test_rel_dist_err[1][1]) if use_jackknife else ''))

        if params['average']=='macro':
            logging.info('Classwise results on unseen test data')
            logging.info('Class\tF\tAE\tdist_err\treldist_err\tSELD_score')
            for cls_cnt in range(params['unique_classes']):
                logging.info('{}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}\t{:0.3f} {}'.format(
                    cls_cnt,

                    classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                    '[{:0.3f}, {:0.3f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                                classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                    '[{:0.3f}, {:0.3f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                                classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                    '[{:0.3f}, {:0.3f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                                classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                    classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                    '[{:0.3f}, {:0.3f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                                classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else '',

                    classwise_test_scr[0][6][cls_cnt] if use_jackknife else classwise_test_scr[6][cls_cnt],
                    '[{:0.3f}, {:0.3f}]'.format(classwise_test_scr[1][6][cls_cnt][0],
                                                classwise_test_scr[1][6][cls_cnt][1]) if use_jackknife else ''))



def get_seld_result(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    score_obj = ComputeSELDResults(params)
    
    dcase_output_val_folder = r'G:\DCASE2024_seld_baseline\results_audio\3_1_dev_split0_multiaccdoa_foa_20240410110923_val'
    val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

    print(val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr)

if __name__ == "__main__":
    
    # get_seld_result(sys.argv)
    try:
        
        main(sys.argv)
    except (ValueError, IOError) as e:
        sys.exit(e)

