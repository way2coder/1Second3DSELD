# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed




class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        
        self.nb_classes = params['unique_classes']
        self.params=params
        self.output_format = params['output_format']
        self.conv_block_list = nn.ModuleList()
        # CNN module list 
        if len(params['f_pool_size']): 
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features = int(in_vid_feat_shape[2]*in_vid_feat_shape[3]), out_features = self.params['rnn_size'] )
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        if self.output_format == 'polar':
            self.localization_output = PolarLocalizationOutput(128, self.params['unique_classes'])
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        # x.shape 128, 7, 250, 64 vid_feat.shape 
        # breakpoint()
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        # x.shape batchsize, 64, 50, 2
        x = x.transpose(1, 2).contiguous() # barchsize 50 64 2 
        x = x.view(x.shape[0], x.shape[1], -1).contiguous() # 128 50 128 
        (x, _) = self.gru(x) # 128 50 256 
        x = torch.tanh(x) # 
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2] # element-wise multipucation to reduce dimision, integrate information, enhance non-linearity

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x # b 50 256 
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in) 
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1): # linear(128, 128) linear(128, 156)
            x = self.fnn_list[fnn_cnt](x)

        if self.output_format == 'multi_accdoa':
            doa = self.fnn_list[-1](x) #x.shape  b,50,128 -> doa.shape ([128, 50, 156])  1.
            doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, self.nb_classes) # b 50 3 4 13  2 
            doa1 = doa[:, :, :, :3, :]  #[128, 50, 3, 3, 13]
            dist = doa[:, :, :, 3:, :]  #[128, 50, 3, 1, 13]

            doa1 = self.doa_act(doa1)
            dist = self.dist_act(dist)
            doa2 = torch.cat((doa1, dist), dim=3)
            doa2 = doa2.reshape((doa.size(0), doa.size(1), -1)) # multiaccdoa: torch.Size([128, 50, 3, 4, 13]) 
        elif self.output_format == 'polar':
            doa2 = self.localization_output(x)   #  b,50,128 -> b 50 13, 4
        return doa2



class PolarLocalizationOutput(nn.Module):

    def __init__(self, input_dim: int, num_classes: int):
        super(PolarLocalizationOutput, self).__init__()

        self.source_activity_output = nn.Linear(input_dim, num_classes)
        self.azimuth_output = nn.Linear(input_dim, num_classes)
        self.elevation_output = nn.Linear(input_dim, num_classes)
        self.distance_output = nn.Linear(input_dim, num_classes)

        self.sed_act = nn.Sigmoid()
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, input: torch.Tensor):
        source_activity = self.source_activity_output(input)
        source_activity = self.sed_act(source_activity)

        azimuth = self.azimuth_output(input)
        azimuth = self.doa_act(azimuth)

        elevation = self.elevation_output(input)
        elevation = self.doa_act(elevation)

        distance = self.distance_output(input)
        distance = self.dist_act(distance)
        
        direction_of_arrival = torch.cat(
            (source_activity.unsqueeze(-1), azimuth.unsqueeze(-1), elevation.unsqueeze(-1), distance.unsqueeze(-1)), dim=-1
        )

    

        return direction_of_arrival
# label target:  polar 128, 50, 52

if __name__ == '__main__':
    tensor1 = torch.rand(3,3)
    tensor2 = torch.rand(3,3)
    tensor3 = torch.cat( (tensor1.unsqueeze(-1), tensor2), dim=-1)
    print(tensor3.shape)

    
