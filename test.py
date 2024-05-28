
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
from cls_dataset import SELDDataset

def main(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = get_params(task_id)
    dataset = SELDDataset(params=params,root='')
    for item in dataset:
        feat, csv = item

if __name__ == '__main__':
    main(sys.argv)