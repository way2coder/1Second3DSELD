# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed
import sys
import parameters
from pytorch_tcn import TCN
from .architecture.CST_former_model import *
import torchaudio
from torchvision import models
import torch.nn.init as init
from efficientnet_pytorch import EfficientNet

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class TCN_CST(torch.nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params=params
        self.conv_block_list = nn.ModuleList()
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

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        self.TCN = TCN(
            num_inputs=20,  # Number of input channels
            num_channels= [64, 128, 256 ,128, 20],  # Number of channels in each residual block
            kernel_size=4,  # Convolution kernel size
            dilations=None,  # Automatic dilation pattern (2^n)
            dilation_reset=16,  # Reset dilation at 16 to manage memory usage
            dropout=0.1,  # Dropout rate
            causal=True,  # Causal convolutions for real-time prediction
            use_norm='weight_norm',  # Weight normalization
            activation='relu',  # Activation function
            kernel_initializer='xavier_uniform',  # Weight initializer
            input_shape='NCL'  # Input shape convention (batch_size, channels, length)
        )

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        # encoder: Conv
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)        

        breakpoint()
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        
        x,  = self.TCN(x)
        x = torch.tanh(x)

        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)
        
        # the below-commented code applies tanh for doa and relu for distance estimates respectively in multi-accdoa scenarios.
        # they can be uncommented and used, but there is no significant changes in the results.
        doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        doa1 = doa[:, :, :, :3, :]
        dist = doa[:, :, :, 3:, :]

        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        return doa2
        # return doa





class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class GSELD(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        self._feature_extraction = params['feature_extraction']
        self._se_block = params['use_se_block']
        self._pe = params['positoncal_encoding']
        self._learnable_pe = params['learnable_pe']
        self._residual_connections = params['residual_connections']
        self._mhsa = params['mhsa']
        self._scconv = params['scconv']

        # ResNet feature extractor
        if self._feature_extraction == 'conv':
            self.conv_block_list = nn.ModuleList()
            if len(params['f_pool_size']):
                for conv_cnt in range(len(params['f_pool_size'])):
                    in_channels = params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1]
                    out_channels = params['nb_cnn2d_filt']
                    self.conv_block_list.append(ConvBlock(in_channels=in_channels, out_channels=out_channels))
                    self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                    self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))

                    # if self._se_block:
                    #     self.conv_block_list.append(SE_Block(c=out_channels, r=16))


        # elif self._feature_extraction == 'resnet':
        #     self.encoder = CustomResNet18WithSE(in_feat_shape=in_feat_shape, params=params)
        # elif self._feature_extraction == 'resnet':
        #     pass
        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)
        
        # self.resnet = resnet18(weights=None)
        # self.resnet.conv1 = nn.Conv2d(in_feat_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # breakpoint()
        # Calculate the output shape of ResNet
        if self._feature_extraction == 'conv':
            with torch.no_grad():
                x = torch.zeros(1, *in_feat_shape[1:])
                for conv_cnt in range(len(self.conv_block_list)):
                    x = self.conv_block_list[conv_cnt](x)

                batch_size, channels, time, freq = x.shape
                x = x.transpose(1, 2).contiguous()
                x = x.view(x.shape[0], x.shape[1], -1).contiguous()

                (x, _) = self.gru(x)  # RuntimeError: Given normalized_shape=[512], expected input with shape [*, 512], but got input of size[20, 1, 256]
                lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)
                
                # breakpoint()
                resnet_out_shape = x.shape[1:]
        else:
            # breakpoint()
            resnet_out_shape = torch.Size([20, params['rnn_size']* 2])
        
        # Conformer layers
        # self.conformer_layers = nn.ModuleList([
        #     torchaudio.models.Conformer(
        #         input_dim= resnet_out_shape[-1],
        #         num_heads=params['cf_num_heads'],  # 可以根据需要调整, 4
        #         ffn_dim=params['cf_ffn_dim'],  # 可以根据要调整, 512, 
        #         num_layers=params['cf_num_layers'],  # 可以根据需要调整,2 
        #         depthwise_conv_kernel_size=params['cf_depthwise_conv_kernel_size'], # 31 
        #     )
        #     for _ in range(params['cf_num_cfs']) # 2 
        # ])

        if self._mhsa is True:
            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
            for mhsa_cnt in range(params['nb_self_attn_layers']):
                self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'] * 2, num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
                self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']* 2 ))
        
        self.TCN = TCN(
            num_inputs=20,  # Number of input channels
            num_channels= params['tcn_channels'],  # Number of channels in each residual block
            kernel_size=3,  # Convolution kernel size
            dilations=None,  # Automatic dilation pattern (2^n)
            dilation_reset=16,  # Reset dilation at 16 to manage memory usage
            dropout=0.1,  # Dropout rate
            causal=True,  # Causal convolutions for real-time prediction
            use_norm='weight_norm',  # Weight normalization
            activation='relu',  # Activation function
            kernel_initializer='xavier_uniform',  # Weight initializer
            input_shape='NCL'  # Input shape convention (batch_size, channels, length)
        )

        
        # Time pooling
        self.time_pool = nn.AdaptiveAvgPool1d(out_shape[1])
        
        # Output layer
        self.output = nn.Linear(resnet_out_shape[-1], out_shape[2])
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # we use xavier_uniform following official JAX ViT:
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        init.zeros_(param.data)


    def forward(self, x):
        # ResNet feature extraction
        # x.shape = in_feat_shape=(128, 7, 100, 128)

        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        batch_size, channels, time, freq = x.shape

        # breakpoint()
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        x = self.TCN(x)
        (x, _) = self.gru(x)  # RuntimeError: Given normalized_shape=[512], expected input with shape [*, 512], but got input of size[20, 1, 256]
        lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)

        # for layer in self.conformer_layers: 
        #     x, _ = layer(x, lengths)   


        # Time pooling
        x = x.transpose(1, 2)
        x = self.time_pool(x)
        x = x.transpose(1, 2)
        
        # Output layer
        x = self.output(x)
        doa = x.reshape(x.shape[0], x.shape[1], 3, 4, 13)

        doa1 = doa[:, :, :, :3, :]
        dist = doa[:, :, :, 3:, :]

        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        return x





if __name__ == '__main__':
    # 初始化模型


    # 输出结果
    # x = torch.rand(128, 100, 45)  
    # avg_pool = nn.AvgPool1d(kernel_size=5, stride=5)  # 根据需要调整kernel_size和stride
    # x_pooled = avg_pool(x.transpose(1, 2)).transpose(1, 2)  # transpose是因为nn.AvgPool1d默认作用于最后一个维度
    # print(x_pooled.shape)
    # try:
    #     sys.exit(main(sys.argv))
    # except (ValueError, IOError) as e:
    #     sys.exit(e)
    pass

    