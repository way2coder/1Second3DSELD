#
# Dataset for training the SELDnet
#

import os
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random
import torch 
from torch.utils.data import Dataset, DataLoader


class STARSS23Dataset(Dataset):
    def __init__(self, params, split, is_eval=False, modality='audio'):
        self._is_eval = is_eval
        self._splits = np.array(split)
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()
        self._modality = modality

        self._filenames_list = []
        self._get_filenames_list()

        if self._modality == 'audio_visual':
            self._vid_feat_dir = self._feat_cls.get_vid_feat_dir()

        print(f'Dataset initialized with {len(self._filenames_list)} files.')