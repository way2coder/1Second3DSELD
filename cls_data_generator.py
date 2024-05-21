#
# Data generator for training the SELDnet
#

import os
import sys
import numpy as np
import cls_feature_class
from IPython import embed
from collections import deque
import random
import parameters

class DataGenerator(object):
    def __init__(
            self, params, split=1, shuffle=True, per_file=False, is_eval=False
    ):
        '''
        per_file : decide whether one file will be trained in batch 
        '''
        self._per_file = per_file   
        self._is_eval = is_eval
        self._splits = np.array(split)    #[1 ,2, 3] actually is [3]
        self._batch_size = params['batch_size'] # 128    
        self._feature_seq_len = params['feature_sequence_length'] #  250 params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution # 50 * 5 
        # feature_label_resolution
        # self.hop
        self._label_seq_len = params['label_sequence_length']  # 50 
        self._shuffle = shuffle
        self._feat_cls = cls_feature_class.FeatureClass(params=params, is_eval=self._is_eval)
        self._label_dir = self._feat_cls.get_label_dir()  # '../Dataset/STARSS2023/feat_label_hnet/foa_dev_multi_accdoa_label'
        self._feat_dir = self._feat_cls.get_normalized_feat_dir()  # '../Dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_norm'
        # self._multi_accdoa = params['multi_accdoa']
        self._output_format = params['output_format']

        self._filenames_list = list()
        self._nb_frames_file = 0     # Using a fixed number of frames in feat files. Updated in _get_label_filenames_sizes()
        self._nb_mel_bins = self._feat_cls.get_nb_mel_bins() # 62 
        self._nb_ch = None
        self._label_len = None  # total length of label - DOA + SED
        self._doa_len = None    # DOA label length 
        self._nb_classes = self._feat_cls.get_nb_classes()  # 13 

        self._circ_buf_feat = None
        self._circ_buf_label = None

        self._modality = params['modality']
        if self._modality == 'audio_visual':
            self._vid_feature_seq_len = self._label_seq_len  # video feat also at 10 fps same as label resolutions (100ms)
            self._vid_feat_dir = self._feat_cls.get_vid_feat_dir()
            self._circ_buf_vid_feat = None

        self._get_filenames_list_and_feat_label_sizes()

        print(
            '\tDatagen_mode: {}, nb_files: {}, nb_classes:{}\n'
            '\tnb_frames_file: {}, feat_len: {}, nb_ch: {}, label_len:{}\n'.format(
                'eval' if self._is_eval else 'dev', len(self._filenames_list),  self._nb_classes,
                self._nb_frames_file, self._nb_mel_bins, self._nb_ch, self._label_len
                )
        )

        print(
            '\tDataset: {}, split: {}\n'
            '\tbatch_size: {}, feat_seq_len: {}, label_seq_len: {}, shuffle: {}\n'
            '\tTotal batches in dataset: {}\n'
            '\tlabel_dir: {}\n '
            '\tfeat_dir: {}\n'.format(
                params['dataset'], split,
                self._batch_size, self._feature_seq_len, self._label_seq_len, self._shuffle,
                self._nb_total_batches,
                self._label_dir, self._feat_dir
            )
        )

    def get_data_sizes(self):
        feat_shape = (self._batch_size, self._nb_ch, self._feature_seq_len, self._nb_mel_bins)
        if self._is_eval:
            label_shape = None
        else:
            if self._output_format == 'multi_accdoa':
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*3*4)
            else:
                label_shape = (self._batch_size, self._label_seq_len, self._nb_classes*4)

        if self._modality == 'audio_visual':
            vid_feat_shape = (self._batch_size, self._vid_feature_seq_len, 7, 7)
            return feat_shape, vid_feat_shape, label_shape
        return feat_shape, label_shape

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def _get_filenames_list_and_feat_label_sizes(self):
        print('Computing some stats about the dataset')   
        max_frames, total_frames, temp_feat = -1, 0, []   
        for filename in os.listdir(self._feat_dir):  #'../Dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_norm'
            if int(filename[4]) in self._splits:  # check which split the file belongs to fold3/ fold4
                if self._modality == 'audio' or (hasattr(self, '_vid_feat_dir') and os.path.exists(os.path.join(self._vid_feat_dir, filename))):   # some audio files do not have corresponding videos. Ignore them.
                    self._filenames_list.append(filename)
                    temp_feat = np.load(os.path.join(self._feat_dir, filename))  # load npy from ''../Dataset/STARSS2023/feat_label_hnet/foa_dev_gammatone_norm\\fold3_room12_mix001.npy'
                    total_frames += (temp_feat.shape[0] - (temp_feat.shape[0] % self._feature_seq_len)) # temp_feat (12664, 448) % 250(input size for neural network)
                    # 12664 - (122664 % 250 = 164 ) = 12500 
                    if temp_feat.shape[0]>max_frames:
                        max_frames = temp_feat.shape[0] # restore the max frame for the spectrum 

        if len(temp_feat)!=0:
            self._nb_frames_file = max_frames if self._per_file else temp_feat.shape[0]  # 10693, 19430
            self._nb_ch = temp_feat.shape[1] // self._nb_mel_bins # 448 / 64 = 7
        else:
            print('Loading features failed')
            exit()

        if not self._is_eval:
            temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[0]))
            # '../Dataset/STARSS2023/feat_label_hnet/foa_dev_multi_accdoa_label\\fold3_room12_mix001.npy'
            # (2532, 6, 5, 13)
            if self._output_format == 'multi_accdoa':
                self._num_track_dummy = temp_label.shape[-3]
                self._num_axis = temp_label.shape[-2]  
                self._num_class = temp_label.shape[-1]
            else:  # single_accoda or polar
                self._label_len = temp_label.shape[-1]  # (2532, 65)
            self._doa_len = 3 # Cartesian

        if self._per_file:
            self._batch_size = int(np.ceil(max_frames/float(self._feature_seq_len))) # 19430 / 250 
            print('\tWARNING: Resetting batch size to {}. To accommodate the inference of longest file of {} frames in a single batch'.format(self._batch_size, max_frames))
            self._nb_total_batches = len(self._filenames_list)
        else:
            self._nb_total_batches = int(np.floor(total_frames / (self._batch_size*self._feature_seq_len))) 
            # 735500 / 32000 = 22 
 
        self._feature_batch_seq_len = self._batch_size*self._feature_seq_len  # 32000 = 128 * 250 
        self._label_batch_seq_len = self._batch_size*self._label_seq_len # 6400 = 128 * 50
        # params['audio_visual'] used to determine wether should calculate the vid_feature_batch_seq_len 
        if self._modality == 'audio_visual':
            self._vid_feature_batch_seq_len = self._batch_size*self._vid_feature_seq_len

        return

    def generate(self):
        """
        Generates batches of samples
        :return: 
        """
        if self._shuffle:
            random.shuffle(self._filenames_list) # ['fold3_*_*.npy']

        # Ideally this should have been outside the while loop. But while generating the test data we want the data
        # to be the same exactly for all epoch's hence we keep it here.
        # breakpoint()
        self._circ_buf_feat = deque()
        self._circ_buf_label = deque()

        if self._modality == 'audio_visual':
            self._circ_buf_vid_feat = deque()

        file_cnt = 0
        if self._is_eval:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it.
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt]))

                    for row_cnt, row in enumerate(temp_feat):
                        self._circ_buf_feat.append(row)

                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))
                        for vf_row_cnt, vf_row in enumerate(temp_vid_feat):
                            self._circ_buf_vid_feat.append(vf_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((extra_frames, temp_feat.shape[1])) * 1e-6

                        for row_cnt, row in enumerate(extra_feat):
                            self._circ_buf_feat.append(row)

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones((vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                            for vf_row_cnt, vf_row in enumerate(extra_vid_feat):
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt = file_cnt + 1

                # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len):
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins))

                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len)
                feat = np.transpose(feat, (0, 2, 1, 3))

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)

                    yield feat, vid_feat
                else:
                    yield feat

        else:
            for i in range(self._nb_total_batches):
                # load feat and label to circular buffer. Always maintain atleast one batch worth feat and label in the
                # circular buffer. If not keep refilling it. self._feature_batch_seq_len = batchsize * T 
                while (len(self._circ_buf_feat) < self._feature_batch_seq_len or (hasattr(self, '_circ_buf_vid_feat') and hasattr(self, '_vid_feature_batch_seq_len') and len(self._circ_buf_vid_feat) < self._vid_feature_batch_seq_len)):
                    temp_feat = np.load(os.path.join(self._feat_dir, self._filenames_list[file_cnt])) # (6500, 448) -> (timestep ,64*7)
                    temp_label = np.load(os.path.join(self._label_dir, self._filenames_list[file_cnt])) # singleaccdoa (1300, 65) -> (timestep/5, 13*5) multiaccdoa (1095, 6, 5, 13)
                    if self._modality == 'audio_visual':
                        temp_vid_feat = np.load(os.path.join(self._vid_feat_dir, self._filenames_list[file_cnt]))

                    if not self._per_file:
                        # In order to support variable length features, and labels of different resolution.
                        # We remove all frames in features and labels matrix that are outside
                        # the multiple of self._label_seq_len and self._feature_seq_len. Further we do this only in training.
                        temp_label = temp_label[:temp_label.shape[0] - (temp_label.shape[0] % self._label_seq_len)]
                        temp_mul = temp_label.shape[0] // self._label_seq_len # 1300 / 50 = 26 
                        temp_feat = temp_feat[:temp_mul * self._feature_seq_len, :] # [:26 * 250 ]
                        if self._modality == 'audio_visual':
                            temp_vid_feat = temp_vid_feat[:temp_mul * self._vid_feature_seq_len, :, :]

                    for f_row in temp_feat:
                        self._circ_buf_feat.append(f_row)   # 6250 -> 17500 
                    for l_row in temp_label:
                        self._circ_buf_label.append(l_row) # 1250 -> 3500

                    if self._modality == 'audio_visual':
                        for vf_row in temp_vid_feat:
                            self._circ_buf_vid_feat.append(vf_row)

                    # If self._per_file is True, this returns the sequences belonging to a single audio recording
                    if self._per_file:
                        feat_extra_frames = self._feature_batch_seq_len - temp_feat.shape[0]
                        extra_feat = np.ones((feat_extra_frames, temp_feat.shape[1])) * 1e-6

                        if self._modality == 'audio_visual':
                            vid_feat_extra_frames = self._vid_feature_batch_seq_len - temp_vid_feat.shape[0]
                            extra_vid_feat = np.ones(
                                (vid_feat_extra_frames, temp_vid_feat.shape[1], temp_vid_feat.shape[2])) * 1e-6

                        label_extra_frames = self._label_batch_seq_len - temp_label.shape[0]
                        if self._output_format == 'multi_accdoa':
                            extra_labels = np.zeros(
                                (label_extra_frames, self._num_track_dummy, self._num_axis, self._num_class))
                        else:
                            extra_labels = np.zeros((label_extra_frames, temp_label.shape[1]))

                        for f_row in extra_feat:
                            self._circ_buf_feat.append(f_row)
                        for l_row in extra_labels:
                            self._circ_buf_label.append(l_row)
                        if self._modality == 'audio_visual':
                            for vf_row in extra_vid_feat:
                                self._circ_buf_vid_feat.append(vf_row)

                    file_cnt = file_cnt + 1

                    # Read one batch size from the circular buffer
                feat = np.zeros((self._feature_batch_seq_len, self._nb_mel_bins * self._nb_ch))
                for j in range(self._feature_batch_seq_len): # 32000 = 128 * 250 
                    feat[j, :] = self._circ_buf_feat.popleft()
                feat = np.reshape(feat, (self._feature_batch_seq_len, self._nb_ch, self._nb_mel_bins)) #  32000, 7, 64

                if self._modality == 'audio_visual':
                    vid_feat = np.zeros((self._vid_feature_batch_seq_len, 7, 7))
                    for v in range(self._vid_feature_batch_seq_len):
                        vid_feat[v, :, :] = self._circ_buf_vid_feat.popleft()

                if self._output_format == 'multi_accdoa':
                    label = np.zeros(
                        (self._label_batch_seq_len, self._num_track_dummy, self._num_axis, self._num_class)) # 128 * 50= 6400, 6, 5, 13 
                    for j in range(self._label_batch_seq_len):
                        label[j, :, :, :] = self._circ_buf_label.popleft()
                elif self._output_format == 'single_accdoa' or self._output_format == 'polar': # TODO: the size of label need to be changed? 
                    label = np.zeros((self._label_batch_seq_len, self._label_len))
                    for j in range(self._label_batch_seq_len):
                        label[j, :] = self._circ_buf_label.popleft()
                # Split to sequences
                feat = self._split_in_seqs(feat, self._feature_seq_len) # 32700, 7, 64 - > (128, 250, 7, 64)
                feat = np.transpose(feat, (0, 2, 1, 3))
                if self._modality == 'audio_visual':
                    vid_feat = self._vid_feat_split_in_seqs(vid_feat, self._vid_feature_seq_len)
 
                label = self._split_in_seqs(label, self._label_seq_len) # multiaccdoa (6400, 6, 5, 13)-> (128, 50, 6, 5, 13)   # 6400, 65, -> 128, 50, 65
                if self._output_format == 'multi_accdoa':
                    pass
                elif self._output_format == 'single_accdoa':  
                    # TODO: polar
                    mask = label[:, :, :self._nb_classes]  # 
                    mask = np.tile(mask, 4)   # 
                    label = mask * label[:, :, self._nb_classes:]#   128, 50, 65->  (128, 50, 52)
                elif self._output_format == 'polar':
                    pass

                if self._modality == 'audio_visual':
                    yield feat, vid_feat, label
                else:
                    yield feat, label     # multiaccdoa:(128, 50, 6, 5, 13)  single  (128, 50, 52)

    def _split_in_seqs(self, data, _seq_len): # data - 250*8, 7, 64 - 250
        if len(data.shape) == 1:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, 1))
        elif len(data.shape) == 2:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1]))
        elif len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        elif len(data.shape) == 4:  # for multi-ACCDOA with ADPIT
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :, :]
            data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2], data.shape[3]))
        else:
            print('ERROR: Unknown data dimensions: {}'.format(data.shape))
            exit()
        return data

    def _vid_feat_split_in_seqs(self, data, _seq_len):
        if len(data.shape) == 3:
            if data.shape[0] % _seq_len:
                data = data[:-(data.shape[0] % _seq_len), :, :]
            else:
                data = data.reshape((data.shape[0] // _seq_len, _seq_len, data.shape[1], data.shape[2]))
        else:
            print('ERROR: Unknown data dimensions for video features: {}'.format(data.shape))
            exit()
        return data

    @staticmethod
    def split_multi_channels(data, num_channels):
        tmp = None
        in_shape = data.shape
        if len(in_shape) == 3:
            hop = in_shape[2] / num_channels
            tmp = np.zeros((in_shape[0], num_channels, in_shape[1], hop))
            for i in range(num_channels):
                tmp[:, i, :, :] = data[:, :, i * hop:(i + 1) * hop]
        elif len(in_shape) == 4 and num_channels == 1:
            tmp = np.zeros((in_shape[0], 1, in_shape[1], in_shape[2], in_shape[3]))
            tmp[:, 0, :, :, :] = data
        else:
            print('ERROR: The input should be a 3D matrix but it seems to have dimensions: {}'.format(in_shape))
            exit()
        return tmp

    def get_nb_classes(self):
        return self._nb_classes

    def nb_frames_1s(self):
        return self._feat_cls.nb_frames_1s()

    def get_hop_len_sec(self):
        return self._feat_cls.get_hop_len_sec()

    def get_filelist(self):
        return self._filenames_list

    def get_frame_per_file(self):
        return self._label_batch_seq_len

    def get_nb_frames(self):
        return self._feat_cls.get_nb_frames()
    
    def get_data_gen_mode(self):
        return self._is_eval

    def write_output_format_file(self, _out_file, _out_dict, _output_format):
        return self._feat_cls.write_output_format_file(_out_file, _out_dict, self._output_format)

def main(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)
    test_dataloader = DataGenerator(params=params, split=[3], shuffle=True)
    i = 0
    for data in test_dataloader.generate():
        i += 1   # feat (128, 7, 250, 64) (barchsize, channel, time, freq) label (128, 50, 6, 5, 13)   (barchsize, time, multi, (xyz,sed,dis), class)

if __name__ == '__main__':
    main(sys.argv)