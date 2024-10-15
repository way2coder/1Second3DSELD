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
from pytorch_tcn import TCN

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

# def update_metrics_history(metrics_history, val_F, val_LE, val_rel_dist_err):
#     """
#     将当前 epoch 的指标添加到历史指标中
#     :param metrics_history: 历史指标的 NumPy 数组
#     :param val_F: 当前 epoch 的 F-score
#     :param val_LE: 当前 epoch 的 DOA 角度误差
#     :param val_rel_dist_err: 当前 epoch 的相对距离误差
#     :return: 更新后的历史指标 NumPy 数组
#     """
#     new_metrics = np.array([[val_F, val_LE, val_rel_dist_err]])
#     return np.vstack([metrics_history, new_metrics])

# def compute_cumulative_rank(metrics_history):
#     """
#     根据历史指标计算每个 epoch 的累计排名
#     :param metrics_history: 历史指标的 NumPy 数组
#     :return: 每个 epoch 的累计排名
#     """
#     ranks = np.empty_like(metrics_history)
    
#     # F-score 越大越好，所以取负数后排序
#     ranks[:, 0] = np.argsort(np.argsort(-metrics_history[:, 0]))
#     # DOA 角度误差越小越好
#     ranks[:, 1] = np.argsort(np.argsort(metrics_history[:, 1]))
#     # 相对距离误差越小越好
#     ranks[:, 2] = np.argsort(np.argsort(metrics_history[:, 2]))
    
#     # 累积排名
#     cumulative_ranks = np.sum(ranks, axis=1)
    
#     return cumulative_ranks

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

    # Example input tensor shape: (batch_size, 3, 128)
    # model = TCN(
    #     num_inputs=20,  # Number of input channels
    #     num_channels= [64, 128, 256 ,128, 20],  # Number of channels in each residual block
    #     kernel_size=4,  # Convolution kernel size
    #     dilations=None,  # Automatic dilation pattern (2^n)
    #     dilation_reset=16,  # Reset dilation at 16 to manage memory usage
    #     dropout=0.1,  # Dropout rate
    #     causal=True,  # Causal convolutions for real-time prediction
    #     use_norm='weight_norm',  # Weight normalization
    #     activation='relu',  # Activation function
    #     kernel_initializer='xavier_uniform',  # Weight initializer
    #     input_shape='NCL'  # Input shape convention (batch_size, channels, length)
    # )

    # Continue training or use the model as needed

    
    # Unique name for the run
    # Collect i/o data size and load model configuration, load model weights to model
    # model we used are wrapped in models/ folder, you can modify it in parameter.py  
    # breakpoint()
    # train_dataset = SELDDataset(params, 'train')
    # total_params = sum(p.numel() for p in model.parameters())
    breakpoint()
    # print(total_params)
    data_in = (1, 7, 100, 128)
    # data_in = (570, 20, 256)
    data_out =(1, 20, 156)  
    model = models[params['model']](data_in, data_out, params)

    input_tensor = torch.randn(data_in)
    output_tensor = model(input_tensor)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model output shape:", output_tensor.shape)







# if __name__ == '__main__':
    # x       = torch.randn(1,7,100,128)
    # model   = ScConv(7)
    # print(model(x).shape)
if __name__ == '__main__':
    main(sys.argv)