# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed
from parameters import get_params
import sys
from torchaudio.models import Conformer


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldConModel(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        
        self.nb_classes = params['unique_classes']
        self.params=params
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

        self.conformer = Conformer(
            input_dim=128,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
        )

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

        self.doa_act = nn.Tanh()
        self.dist_linear = nn.Linear(self.nb_classes, self.nb_classes)
        self.dist_act = nn.ELU()

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        # x.shape 128, 7, 250, 64 vid_feat.shape 
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        # x.shape batchsize, 64, 50, 2
        x = x.transpose(1, 2).contiguous() # barchsize 50 64 2 
        x = x.view(x.shape[0], x.shape[1], -1).contiguous() # 128 50 128 
        (x, _) = self.gru(x) # 128 50 256 
        x = torch.tanh(x) # 
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2] # element-wise multipucation to reduce dimision, integrate information, enhance non-linearity
        

        lengths = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
        output = self.conformer(x , lengths)
        x, _ = output

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x # b 50 256 
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in) 
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)
        
        # breakpoint()
        for fnn_cnt in range(len(self.fnn_list) - 1): # linear(128, 128) linear(128, 156)
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x) # b,50,156

        doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, self.nb_classes) # b 50 3 4 13
        doa1 = doa[:, :, :, :3, :]  #[128, 50, 3, 3, 13]
        dist = doa[:, :, :, 3:, :]  #[128, 50, 3, 1, 13]

        dist = self.dist_linear(dist)


        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        
        return doa2




if __name__ == '__main__':
    # params = get_params(sys.argv)
    conformer = Conformer(
        input_dim=128,
        num_heads=4,
        ffn_dim=128,
        num_layers=4,
        depthwise_conv_kernel_size=31,
    )
    lengths = torch.full((128,), 50, dtype=torch.long)
    input = torch.rand(128, 50, 128) 
    
    output = conformer(input , lengths)

    frames, lengths = output
    print(frames.shape)
    print(lengths.shape)
    # model = SeldModel((128,7,100,64),(128,50,168),params=params)
