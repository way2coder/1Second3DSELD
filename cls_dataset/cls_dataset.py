import hashlib
import json
import numpy as np
import os
import pandas as pd
from scipy.io import wavfile
from scipy.signal import stft
from torch.utils.data import Dataset
from typing import Tuple
import torch
from cls_feature_class import FeatureClass
from parameters import get_params
import sys

from abc import ABC, abstractmethod
from collections import OrderedDict



class LRUCache:
    def __init__(self, capacity: int = 10) -> None:
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None 
        else: 
            self.cache.move_to_end(key)
            return self.cache[key]
        
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class SELDDataset(Dataset):

    def __init__(
        self,
        params: dict,
        train_test: str = 'train', #'test' val 
        splits : list = [1, 2, 3],
        overlapping_events: list = [1,2,3],
        is_eval: bool = False
    ) -> None:
        # self._per_file = per_file   
        self._is_eval = is_eval
        # self._splits = np.array(split)    #[1 ,2, 3] actually is [3] 
        self._dataset = params['dataset']          
        self._train_test = train_test
        self._splits = splits
        self._overlapping_events = overlapping_events


        self._feature_seq_len = params['feature_sequence_length'] #  250 params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution # 50 * 5 
        # feature_label_resolution
        # self.hop
        self._label_seq_len = params['label_sequence_length']  # 50 
        self._feat_cls = FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()  # '../Dataset/STARSS2023/feat_label_hnet/foa_dev_multi_accdoa_label'
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()  # '../Dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_norm'
        # self._multi_accdoa = params['multi_accdoa']
        self._output_format = params['output_format']

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins() # 64
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length 
        self._nb_classes = self._feat_cls.get_nb_classes()  # 14


        self.cache = LRUCache(capacity=10)
        self._circ_buf_feat = None
        self._circ_buf_label = None

        self.chunks = {}
        # breakpoint()
        self._feat_cls.get_frame_stats() #  

        self._filewise_frames = self._feat_cls._filewise_frames   # all the features and labels of favoured splits are stored in the self.feat_cls._filewise_frames, when assign the train, valid, test.
        # TODO filter the file frame spilts 
        self._filewise_frames = self._filter_filewise_frames(self._filewise_frames, self._dataset, self._train_test)
    
        self._get_chunks_state_dict(self._feat_dir, self._label_dir, train_test=self._train_test,splits=self._splits , overlapping_events=self._overlapping_events)
        
    def _filter_filewise_frames(self, filewise_names, dataset, train_test):
        # breakpoint()
        filtered_dict = {}
        if dataset in ['STARSS2023']:
            if train_test == 'train':
                for key, value in filewise_names.items():
                    # 只添加不包含'fold4'的键
                    if 'fold3' in key:
                        filtered_dict[key] = value
            elif train_test == 'test' or train_test == 'val':
                for key, value in filewise_names.items():
                    # 只添加不包含'fold4'的键
                    if 'fold4' in key:
                        filtered_dict[key] = value
        else:
            pass
        return filtered_dict


    
    def _get_chunks_state_dict(
        self,
        feat_dir: str,
        label_dir: str,
        train_test: list,
        splits :list, 
        overlapping_events: int,
    ) -> None:
        index = 0
        for item in self._filewise_frames.items():
            file_name, (feat_len, label_len) = item 
            file_name += '.npy'
            feat_npy_file = os.path.join(self._feat_dir, file_name)
            label_npy_file = os.path.join(self._label_dir, file_name)

            feat_chunk_len = self._feature_seq_len 
            label_chunk_len = self._label_seq_len
            num_feat_chunks = feat_len  // feat_chunk_len
            num_label_chunks = label_len // label_chunk_len
            # breakpoint()
            assert num_feat_chunks == num_label_chunks, 'num_feat_chunks should equal to num_label_chunks'
            
            for i in range(num_feat_chunks):
                feat_start_frame = i * feat_chunk_len
                feat_end_frame = (i + 1) * feat_chunk_len
                label_start_frame = i * label_chunk_len
                label_end_frame = (i + 1) * label_chunk_len
                self.chunks[index] = {
                    "feat_npy_file": feat_npy_file,
                    "label_npy_file": label_npy_file,
                    "feat_start_frame": feat_start_frame,
                    "feat_end_frame": feat_end_frame,
                    "label_start_frame": label_start_frame,
                    "label_end_frame": label_end_frame,
                }
                index += 1

        
    # 对于每一个self._filewise_frame的文件，也就是key，获取他的feat和label的长度，然后根据self._label_seq_len， self._feature_seq_len将整个文件划分为一个个的chunks，对于文件末尾不够一个chunk的片段，直接丢弃不计入chunk中 
            # self.chunks[sequence_idx] = {
            #     "feat_npy_file": audio_file,
            #     "label_npy_file": annotation_file,
            #     "feat_start_frame":feat_start_frame,
            #     "feat_end_frame": feat_end_frame,
            #     "label_start_frame":feat_start_frame,
            #     "label_end_frame": feat_end_frame,
            # }


    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(
        self, index: int
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        chunk = self.chunks[index]
        feat_file_path = os.path.abspath(chunk['feat_npy_file'])
        label_file_path = os.path.abspath(chunk['label_npy_file'])
        
        feat_npy = self.cache.get(feat_file_path)
        if feat_npy is None:
            feat_npy = np.load(feat_file_path)
            self.cache.put(feat_file_path, feat_npy)
        
        label_npy = self.cache.get(label_file_path)
        if label_npy is None:
            label_npy = np.load(label_file_path)
            self.cache.put(label_file_path, label_npy)

        feat = feat_npy[chunk['feat_start_frame']:chunk['feat_end_frame']]
        label = label_npy[chunk['label_start_frame']:chunk['label_end_frame']]

        feat = feat.reshape(feat.shape[0], 7, self._nb_mel_bins)
        feat = np.transpose(feat, (1, 0, 2))

        feat_tensor = torch.from_numpy(feat).float()
        label_tensor = torch.from_numpy(label).float()
        # feat_npy = np.load(os.path.abspath(chunk['feat_npy_file']))
        # label_npy = np.load(os.path.abspath(chunk["label_npy_file"]))
        

        # feat_npy = self.cache.get()
        # feat_start_frame = chunk['feat_start_frame']
        # feat_end_frame = chunk['feat_end_frame']

        # label_start_frame = chunk['label_start_frame']
        # label_end_frame = chunk['label_end_frame']
        
        # feat = feat_npy[feat_start_frame:feat_end_frame]
        # label = label_npy[label_start_frame:label_end_frame]
        # feat = feat.reshape(feat.shape[0], 7, self._nb_mel_bins)
        # # breakpoint()
    
        # feat = np.transpose(feat, (1, 0, 2))
        # # print(source_activity.shape)
        # feat_tensor = torch.from_numpy(feat).float()
        # label_tensor = torch.from_numpy(label).float()
        
        # 返回特征和标签的张量
        return feat_tensor, label_tensor








# class TUTDataset(SELDDataset, Dataset):

#     def __init__(
#         self,
#         params: dict,
#         root: str,
#         split: str = "train",
#         test_fold_idx: int = 1,
#         frame_length: float = 0.04,
#     ) -> None:
#         """Class constructor.

#         :param root: path to root directory of the desired subset
#         :param split: choose between 'train' (default), 'valid' and 'test'
#         :param test_fold_idx: cross-validation index used for testing; choose between 1, 2, and 3
#         :param sequence_duration: fixed duration of each audio signal in seconds, which is set to 30s by default
#         :param num_fft_bins: number of FFT bins
#         """
#         if not os.path.isdir(tmp_dir):
#             os.makedirs(tmp_dir)
#         self.tmp_dir = tmp_dir

#         if split not in ["train", "valid", "test"]:
#             raise RuntimeError(
#                 "Split must be specified as either train, valid or test."
#             )

#         if (test_fold_idx < 1) or (test_fold_idx > 3):
#             raise RuntimeError(
#                 "The desired test fold index must be specified as either 1, 2 or 3."
#             )
        

#         self.split = split
#         self.test_fold_idx = test_fold_idx
#         self.is_TUT = (os.path.basename(root) != 'L3DAS21') 
#         if os.path.basename(root) == 'L3DAS21': 
#             self.sequence_duration = 60.0
#         else :
#             self.sequence_duration = 30.0
#         self.chunk_length = chunk_length
#         self.num_chunks_per_sequence = int(self.sequence_duration / self.chunk_length)
#         # if frame_length == None:
#         #     self.frame_length = chunk_length / 50
#         # else:
#         self.frame_length = frame_length
#         self.num_fft_bins = num_fft_bins
#         self.max_num_sources = max_num_sources
#         self.num_overlapping_sources = num_overlapping_sources

#         # Assemble table containing paths to all audio and annotation files.
#         self.sequences = {}

#         for audio_subfolder in os.listdir(root):
#                 if os.path.isdir(
#                     os.path.join(root, audio_subfolder)
#                 ) and audio_subfolder.startswith("wav"):
#                     annotation_subfolder = "desc" + audio_subfolder[3:-5]

#                     if num_overlapping_sources is not None:
#                         if num_overlapping_sources != int(
#                             annotation_subfolder[annotation_subfolder.find("ov") + 2]
#                         ):
#                             continue
                    
#                     fold_idx = int(
#                         annotation_subfolder[annotation_subfolder.find("split") + 5]
#                     )
                    
#                     for file in os.listdir(os.path.join(root, audio_subfolder)):
#                         file_prefix, extension = os.path.splitext(file)

#                         if extension == ".wav":
#                             path_to_audio_file = os.path.join(root, audio_subfolder, file)
#                             path_to_annotation_file = os.path.join(
#                                 root, annotation_subfolder, file_prefix + ".csv"
#                             )
#                             is_train_file = file_prefix.startswith("train")

#                             # Check all three possible cases where files will be added to the global file list
#                             # 当 split 参数设置为 'train' 时，所有不属于 test_fold_idx 的 'train' 文件都被包括在训练集中。
#                             # 当 split 参数设置为 'valid' 时，所有不属于 test_fold_idx 的 'valid' 文件都被包括在验证集中。
#                             # 当 split 参数设置为 'test' 时，所有属于 test_fold_idx 的文件都被包括在测试集中。
#                             if (
#                                 (split == "train")
#                                 and (fold_idx != test_fold_idx)
#                                 and is_train_file
#                             ):
#                                 self._append_sequence(
#                                     path_to_audio_file,
#                                     path_to_annotation_file,
#                                     is_train_file,
#                                     fold_idx,
#                                     num_overlapping_sources,
#                                 )
#                             elif (
#                                 (split == "valid")
#                                 and (fold_idx != test_fold_idx)
#                                 and not is_train_file
#                             ):
#                                 self._append_sequence(
#                                     path_to_audio_file,
#                                     path_to_annotation_file,
#                                     is_train_file,
#                                     fold_idx,
#                                     num_overlapping_sources,
#                                 )
#                             elif (split == "test") and (fold_idx == test_fold_idx):
#                                 self._append_sequence(
#                                     path_to_audio_file,
#                                     path_to_annotation_file,
#                                     is_train_file,
#                                     fold_idx,
#                                     num_overlapping_sources,
#                                 )
       
#     def _append_sequence(
#         self,
#         audio_file: str,
#         annotation_file: str,
#         is_train_file: bool,
#         fold_idx: int,
#         num_overlapping_sources: int,
#     ) -> None:
#         """Appends sequence (audio and annotation file) to global list of sequences.

#         :param audio_file: path to audio file
#         :param annotation_file: path to corresponding annotation file in *.csv-format
#         :param is_train_file: flag indicating if file is used for training
#         :param fold_idx: cross-validation fold index of current file
#         :param num_overlapping_sources: number of overlapping sources in the dataset
#         """
#         for chunk_idx in range(self.num_chunks_per_sequence):
#             sequence_idx = len(self.sequences)

#             start_time = chunk_idx * self.chunk_length
#             end_time = start_time + self.chunk_length

#             self.sequences[sequence_idx] = {
#                 "audio_file": audio_file,
#                 "annotation_file": annotation_file,
#                 "is_train_file": is_train_file,
#                 "cv_fold_idx": fold_idx,
#                 "chunk_idx": chunk_idx,
#                 "start_time": start_time,
#                 "end_time": end_time,
#                 "num_overlapping_sources": num_overlapping_sources,
#             }

#     def _get_audio_features(
#         self, audio_file: str, start_time: float = None, end_time: float = None
#     ) -> np.ndarray:
#         """Returns magnitude and phase of the multi-channel spectrogram for a given audio file.

#         :param audio_file: path to audio file
#         :param start_time: start time of the desired chunk in seconds
#         :param end_time: end time of the desired chunk in seconds
#         :return: magnitude, phase and sampling rate in Hz
#         """
#         sampling_rate, audio_data = wavfile.read(audio_file)
#         num_samples, num_channels = audio_data.shape

#         required_num_samples = int(sampling_rate * self.sequence_duration)

#         # Perform zero-padding (if required) or truncate signal if it exceeds the desired duration.
#         if num_samples < required_num_samples:
#             audio_data = np.pad(
#                 audio_data,
#                 ((0, required_num_samples - num_samples), (0, 0)),
#                 mode="constant",
#             )
#         elif num_samples > required_num_samples:
#             audio_data = audio_data[:required_num_samples, :]

#         # Normalize and crop waveform
#         start_time_samples = int(start_time * sampling_rate)
#         end_time_samples = int(end_time * sampling_rate)

#         waveform = audio_data[start_time_samples:end_time_samples, :]
#         waveform = waveform / np.iinfo(waveform.dtype).max

#         # Compute multi-channel STFT and remove first coefficient and last frame
#         frame_length_samples = int(self.frame_length * sampling_rate)
#         spectrogram = stft(
#             waveform,
#             fs=sampling_rate,
#             nperseg=frame_length_samples,
#             nfft=self.num_fft_bins,
#             axis=0,
#         )[-1]
#         spectrogram = spectrogram[1:, :, :-1]
#         spectrogram = np.transpose(spectrogram, [1, 2, 0])

#         # Compose output tensor as concatenated magnitude and phase spectra
#         audio_features = np.concatenate(
#             (np.abs(spectrogram), np.angle(spectrogram)), axis=0
#         )

#         return audio_features.astype(np.float16)

#     def _get_targets(
#         self, annotation_file: str, chunk_start_time: float = None
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Returns a polar map of directions-of-arrival (azimuth and elevation) from a given annotation file.

#         :param annotation_file: path to annotation file
#         :param chunk_start_time: start time of the desired chunk in seconds
#         :return: two-dimensional DoA map
#         """
#         # Check the format by examining the columns
        
#         if self.is_TUT is True:
#             annotations = pd.read_csv(annotation_file, header=0, names=[
#             'sound_event_recording', 'start_time', 'end_time', 'elevation', 'azimuth', 'distance'])
#         else:
#             annotations = pd.read_csv(annotation_file,header=0)
#             annotations['azimuth'], annotations['elevation'] = zip(*annotations.apply(lambda row: cartesian_to_spherical(row['X'], row['Y'], row['Z']), axis=1))
#             annotations.rename(columns={
#                 'Class': 'sound_event_recording',
#                 'Start': 'start_time',
#                 'End': 'end_time',
#                 # 'Class' column is not renamed as it does not directly match any target column names.
#                 # If you need to use 'Class' for something specific, consider how you want to incorporate it.

#                 # 'azimuth' and 'elevation' already match the target naming scheme
#             }, inplace=True)
#             annotations.drop(['File', 'X', 'Y', 'Z'], axis=1, inplace=True)



#         # annotations = pd.read_csv(
#         #     annotation_file,
#         #     header=0,
#         #     names=[
#         #         "sound_event_recording",
#         #         "start_time",
#         #         "end_time",
#         #         "elevation",
#         #         "azimuth",
#         #         "distance",
#         #     ],
#         # )
#         # 
        
#         annotations = annotations.sort_values("start_time")

#         chunk_end_time = chunk_start_time + self.chunk_length
#         event_start_time = annotations["start_time"].to_numpy()
#         event_end_time = annotations["end_time"].to_numpy()

#         num_frames_per_chunk = int(2 * self.chunk_length / self.frame_length)

#         source_activity = np.zeros(
#             (num_frames_per_chunk, self.max_num_sources), dtype=np.uint8
#         )
#         direction_of_arrival = np.zeros(
#             (num_frames_per_chunk, self.max_num_sources, 2), dtype=np.float32
#         )

#         event_mask = event_start_time <= chunk_start_time
#         event_mask = event_mask | (
#             (event_start_time >= chunk_start_time) & (event_start_time < chunk_end_time)
#         )
#         event_mask = event_mask & (event_end_time > chunk_start_time)

#         events_in_chunk = annotations[event_mask].copy()  # annotations:dataframe
#         num_active_sources = len(events_in_chunk)

#         active_event_start_time = events_in_chunk["start_time"].to_numpy()
#         active_event_end_time = events_in_chunk["end_time"].to_numpy()

#         active_event_start_time_in_chunk = np.maximum(
#             active_event_start_time, chunk_start_time
#         )
#         active_event_end_time_in_chunk = np.minimum(
#             active_event_end_time, chunk_end_time
#         )

#         event_in_chunk_start_frame_idx = (
#             (active_event_start_time_in_chunk - chunk_start_time) / self.chunk_length
#         ) * num_frames_per_chunk
#         event_in_chunk_end_frame_idx = (
#             (active_event_end_time_in_chunk - chunk_start_time) / self.chunk_length
#         ) * num_frames_per_chunk

#         event_in_chunk_start_frame_idx = np.ceil(event_in_chunk_start_frame_idx).astype(
#             np.int16
#         )
#         event_in_chunk_end_frame_idx = np.floor(event_in_chunk_end_frame_idx).astype(
#             np.int16
#         )
#         event_in_chunk_duration = (
#             event_in_chunk_end_frame_idx - event_in_chunk_start_frame_idx
#         )
#         events_in_chunk.loc[:, "duration"] = event_in_chunk_duration
#         events_in_chunk.sort_values("duration", ascending=False)
#         # 
#         if num_active_sources > 0:
#             i = 0
#             for _, event in events_in_chunk.iterrows():
#                 source_activity[
#                     event_in_chunk_start_frame_idx[i] : event_in_chunk_end_frame_idx[i],
#                     i,
#                 ] = 1
#                 direction_of_arrival[
#                     event_in_chunk_start_frame_idx[i] : event_in_chunk_end_frame_idx[i],
#                     i,
#                     0,
#                 ] = np.deg2rad(event.azimuth)
#                 direction_of_arrival[
#                     event_in_chunk_start_frame_idx[i] : event_in_chunk_end_frame_idx[i],
#                     i,
#                     1,
#                 ] = np.deg2rad(event.elevation)
#                 i += 1
#                 if i >= self.max_num_sources:
#                     break

#         # for frame_idx in range(num_frames_per_chunk):
#         #     frame_start_time = chunk_start_time + frame_idx * (self.frame_length / 2)
#         #     frame_end_time = frame_start_time + (self.frame_length / 2)

#         #     event_mask = event_start_time <= frame_start_time
#         #     event_mask = event_mask | ((event_start_time >= frame_start_time) & (event_start_time < frame_end_time))
#         #     event_mask = event_mask & (event_end_time > frame_start_time)

#         #     events_in_chunk = annotations[event_mask]
#         #     num_active_sources = len(events_in_chunk)

#         #     if num_active_sources > 0:
#         #         source_activity[frame_idx, :num_active_sources] = 1
#         #         direction_of_arrival[frame_idx, :num_active_sources, :] = np.deg2rad(
#         #             events_in_chunk[['azimuth', 'elevation']].to_numpy())
#         return source_activity, direction_of_arrival

#     def _get_parameter_hash(self) -> str:
#         """Returns a hash value encoding the dataset parameter settings.

#         :return: hash value
#         """
#         parameter_dict = {
#             "chunk_length": self.chunk_length,
#             "frame_length": self.frame_length,
#             "num_fft_bins": self.num_fft_bins,
#             "sequence_duration": self.sequence_duration,
#         }

#         return hashlib.md5(
#             json.dumps(parameter_dict, sort_keys=True).encode("utf-8")
#         ).hexdigest()

#     def __len__(self) -> int:
#         return len(self.sequences)

#     def __getitem__(
#         self, index: int
#     ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
#         sequence = self.sequences[index]

#         file_path, file_name = os.path.split(sequence["audio_file"])
#         group_path, group_name = os.path.split(file_path)
#         _, dataset_name = os.path.split(group_path)
#         parameter_hash = self._get_parameter_hash()

#         feature_file_name = file_name + "_" + str(sequence["chunk_idx"]) + "_f.npz"
#         target_file_name = (
#             file_name
#             + "_"
#             + str(sequence["chunk_idx"])
#             + "_t"
#             + str(self.max_num_sources)
#             + ".npz"
#         )

#         path_to_feature_file = os.path.join(
#             self.tmp_dir, dataset_name, group_name, parameter_hash
#         )
#         if not os.path.isdir(path_to_feature_file):
#             try:
#                 os.makedirs(path_to_feature_file)
#             except:
#                 pass

#         if os.path.isfile(os.path.join(path_to_feature_file, feature_file_name)):
#             data = np.load(
#                 os.path.join(path_to_feature_file, feature_file_name), allow_pickle=True
#             )
#             audio_features = data["audio_features"]
#         else:
#             audio_features = self._get_audio_features(
#                 sequence["audio_file"], sequence["start_time"], sequence["end_time"]
#             )
#             np.savez_compressed(
#                 os.path.join(path_to_feature_file, feature_file_name),
#                 audio_features=audio_features,
#             )

#         if os.path.isfile(os.path.join(path_to_feature_file, target_file_name)):
#             data = np.load(
#                 os.path.join(path_to_feature_file, target_file_name), allow_pickle=True
#             )
#             source_activity = data["source_activity"]
#             direction_of_arrival = data["direction_of_arrival"]
#         else:
#             source_activity, direction_of_arrival = self._get_targets(
#                 sequence["annotation_file"], sequence["start_time"]
#             )
#             np.savez_compressed(
#                 os.path.join(path_to_feature_file, target_file_name),
#                 source_activity=source_activity,
#                 direction_of_arrival=direction_of_arrival,
#             )
#         # print(source_activity.shape)
#         return audio_features.astype(np.float32), (
#             source_activity.astype(np.float32),
#             direction_of_arrival.astype(np.float32),
#         )
# # Function to convert Cartesian to spherical coordinates (azimuth and elevation)

