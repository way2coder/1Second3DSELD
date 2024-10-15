# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed
import sys
import parameters
import torchaudio
from torchvision import models
import torch.nn.init as init
from pytorch_tcn import TCN
from efficientnet_pytorch import EfficientNet

class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num:int, 
                 group_num:int = 16, 
                 eps:float = 1e-10
                 ):
        super(GroupBatchnorm2d,self).__init__()
        assert c_num    >= group_num
        self.group_num  = group_num
        self.weight     = nn.Parameter( torch.randn(c_num, 1, 1)    )
        self.bias       = nn.Parameter( torch.zeros(c_num, 1, 1)    )
        self.eps        = eps
    def forward(self, x):
        N, C, H, W  = x.size()
        x           = x.view(   N, self.group_num, -1   )
        mean        = x.mean(   dim = 2, keepdim = True )
        std         = x.std (   dim = 2, keepdim = True )
        x           = (x - mean) / (std+self.eps)
        x           = x.view(N, C, H, W)
        return x * self.weight + self.bias


class SRU(nn.Module):
    def __init__(self,
                 oup_channels:int, 
                 group_num:int = 16,
                 gate_treshold:float = 0.5,
                 torch_gn:bool = True
                 ):
        super().__init__()
        
        self.gn             = nn.GroupNorm( num_channels = oup_channels, num_groups = group_num ) if torch_gn else GroupBatchnorm2d(c_num = oup_channels, group_num = group_num)
        self.gate_treshold  = gate_treshold
        self.sigomid        = nn.Sigmoid()

    def forward(self,x):
        gn_x        = self.gn(x)
        w_gamma     = self.gn.weight/sum(self.gn.weight)
        w_gamma     = w_gamma.view(1,-1,1,1)
        reweigts    = self.sigomid( gn_x * w_gamma )
        # Gate
        w1          = torch.where(reweigts > self.gate_treshold, torch.ones_like(reweigts), reweigts) # 大于门限值的设为1，否则保留原值
        w2          = torch.where(reweigts > self.gate_treshold, torch.zeros_like(reweigts), reweigts) # 大于门限值的设为0，否则保留原值
        x_1         = w1 * x
        x_2         = w2 * x
        y           = self.reconstruct(x_1,x_2)
        return y
    
    def reconstruct(self,x_1,x_2):
        x_11,x_12 = torch.split(x_1, x_1.size(1)//2, dim=1)
        x_21,x_22 = torch.split(x_2, x_2.size(1)//2, dim=1)
        return torch.cat([ x_11+x_22, x_12+x_21 ],dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''
    def __init__(self, 
                 op_channel:int,
                 alpha:float = 1/2,
                 squeeze_radio:int = 2 ,
                 group_size:int = 2,
                 group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.up_channel     = up_channel   =   int(alpha*op_channel)
        self.low_channel    = low_channel  =   op_channel-up_channel
        self.squeeze1       = nn.Conv2d(up_channel,up_channel//squeeze_radio,kernel_size=1,bias=False)
        self.squeeze2       = nn.Conv2d(low_channel,low_channel//squeeze_radio,kernel_size=1,bias=False)
        #up
        self.GWC            = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=group_kernel_size, stride=1,padding=group_kernel_size//2, groups = group_size)
        self.PWC1           = nn.Conv2d(up_channel//squeeze_radio, op_channel,kernel_size=1, bias=False)
        #low
        self.PWC2           = nn.Conv2d(low_channel//squeeze_radio, op_channel-low_channel//squeeze_radio,kernel_size=1, bias=False)
        self.advavg         = nn.AdaptiveAvgPool2d(1)

    def forward(self,x):
        # Split
        up,low  = torch.split(x,[self.up_channel,self.low_channel],dim=1)
        up,low  = self.squeeze1(up),self.squeeze2(low)
        # Transform
        Y1      = self.GWC(up) + self.PWC1(up)
        Y2      = torch.cat( [self.PWC2(low), low], dim= 1 )
        # Fuse
        out     = torch.cat( [Y1,Y2], dim= 1 )
        out     = F.softmax( self.advavg(out), dim=1 ) * out
        out1,out2 = torch.split(out,out.size(1)//2,dim=1)
        return out1+out2


class ScConv(nn.Module):
    def __init__(self,
                op_channel:int,
                group_num:int = 4,
                gate_treshold:float = 0.5,
                alpha:float = 1/2,
                squeeze_radio:int = 2 ,
                group_size:int = 2,
                group_kernel_size:int = 3,
                 ):
        super().__init__()
        self.SRU = SRU( op_channel, 
                       group_num            = group_num,  
                       gate_treshold        = gate_treshold )
        self.CRU = CRU( op_channel, 
                       alpha                = alpha, 
                       squeeze_radio        = squeeze_radio ,
                       group_size           = group_size ,
                       group_kernel_size    = group_kernel_size )
    
    def forward(self,x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x
    


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x

class SELDConformer(nn.Module):
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
        # Posional Encoding
        if self._scconv is True:
            self.scconv = ScConv(params['nb_cnn2d_filt'])
        
        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            torchaudio.models.Conformer(
                input_dim= resnet_out_shape[-1],
                num_heads=params['cf_num_heads'],  # 可以根据需要调整, 4
                ffn_dim=params['cf_ffn_dim'],  # 可以根据要调整, 512, 
                num_layers=params['cf_num_layers'],  # 可以根据需要调整,2 
                depthwise_conv_kernel_size=params['cf_depthwise_conv_kernel_size'], # 31 
            )
            for _ in range(params['cf_num_cfs']) # 2 
        ])

        if self._mhsa is True:
            self.mhsa_block_list = nn.ModuleList()
            self.layer_norm_list = nn.ModuleList()
            for mhsa_cnt in range(params['nb_self_attn_layers']):
                self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'] * 2, num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
                self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']* 2 ))
        

        
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
        if self._feature_extraction == 'conv':
            for conv_cnt in range(len(self.conv_block_list)):
                x = self.conv_block_list[conv_cnt](x)
        elif self._feature_extraction == 'resnet':
            x = self.encoder(x)
        # breakpoint()
        batch_size, channels, time, freq = x.shape
        if self._scconv is True:
            x = self.scconv(x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        (x, _) = self.gru(x)  # RuntimeError: Given normalized_shape=[512], expected input with shape [*, 512], but got input of size[20, 1, 256]
        lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)
        # breakpoint()

        for layer in self.conformer_layers: 
            x, _ = layer(x, lengths)   


        if self._mhsa is True:
            for mhsa_cnt in range(len(self.mhsa_block_list)):
                x_attn_in = x
                x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
                x = x + x_attn_in
                x = self.layer_norm_list[mhsa_cnt](x)

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




class SELDConformerEdit2(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        self.use_efficientnet = params['use_efficientnet']
        self.use_se_block = params['use_se_block']
        
        # ResNet feature extractor
        if self.use_efficientnet:
            # Pretrained EfficientNet feature extractor
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0',in_channels=7,num_classes=64 * 20 * 4)
            self.feature_extractor._conv_stem = nn.Conv2d(in_feat_shape[1], 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.conv_out_channels = 32
        else:
            # Original ConvBlock feature extractor
            self.conv_block_list = nn.ModuleList()
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))
            self.conv_out_channels = params['nb_cnn2d_filt']


        if self.use_se_block:
            self.se_block = SE_Block(64)

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
        
        # self.resnet = resnet18(weights=None)
        # self.resnet.conv1 = nn.Conv2d(in_feat_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # breakpoint()
        # Calculate the output shape of ResNet
        with torch.no_grad():
            x = torch.zeros(1, *in_feat_shape[1:])
            for conv_cnt in range(len(self.conv_block_list)):
                x = self.conv_block_list[conv_cnt](x)

            x = x.transpose(1, 2).contiguous()
            x = x.view(x.shape[0], x.shape[1], -1).contiguous()
            # x = self.TCN(x)
        
            (x, _) = self.gru(x)
            resnet_out_shape = x.shape[1:] # 20, 512

        # Fully connected layer after ResNet
        self.fc = nn.Linear(resnet_out_shape[0]  * resnet_out_shape[1], 512)
        
        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            torchaudio.models.Conformer(
                input_dim=resnet_out_shape[-1],
                num_heads=params['cf_num_heads'],  # 可以根据需要调整
                ffn_dim=params['cf_ffn_dim'],  # 可以根据需要调整
                num_layers=params['cf_num_layers'],  # 可以根据需要调整
                depthwise_conv_kernel_size=params['cf_depthwise_conv_kernel_size']
            )
            for _ in range(params['cf_num_cfs'])
        ])
        # Time pooling
        self.time_pool = nn.AdaptiveAvgPool1d(out_shape[1])
        
        # Output layer
        self.output = nn.Linear(resnet_out_shape[-1], out_shape[2])
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
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
        breakpoint()
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)
        
        batch_size, channels, time, freq = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        (x, _) = self.gru(x)
        lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)
        
        for layer in self.conformer_layers: 
            x, _ = layer(x, lengths)   
        # Time pooling
        x = x.transpose(1, 2)
        x = self.time_pool(x)
        x = x.transpose(1, 2)
        
        # Output layer
        # breakpoint()
        x = self.output(x)
        doa = x.reshape(x.shape[0], x.shape[1], 3, 4, 13)

        doa1 = doa[:, :, :, :3, :]
        dist = doa[:, :, :, 3:, :]

        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        # breakpoint()
        return x




class SELDConformerEdit(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        self.use_efficientnet = params['use_efficientnet']
        self.use_se_block = params['use_se_block']


        # ResNet feature extractor
       # Feature extractor
        if self.use_efficientnet:
            # Pretrained EfficientNet feature extractor
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0',in_channels=7,num_classes=64 * 20 * 4)
            self.feature_extractor._conv_stem = nn.Conv2d(in_feat_shape[1], 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            self.conv_out_channels = 32
        else:
            # Original ConvBlock feature extractor
            self.conv_block_list = nn.ModuleList()
            for conv_cnt in range(len(params['f_pool_size'])):
                self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
                self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
                self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))
            self.conv_out_channels = params['nb_cnn2d_filt']

            
    
        # self.conv_block_list = nn.ModuleList()
        # if len(params['f_pool_size']):
        #     for conv_cnt in range(len(params['f_pool_size'])):
        #         self.conv_block_list.append(ConvBlock(in_channels=params['nb_cnn2d_filt'] if conv_cnt else in_feat_shape[1], out_channels=params['nb_cnn2d_filt']))
        #         self.conv_block_list.append(nn.MaxPool2d((params['t_pool_size'][conv_cnt], params['f_pool_size'][conv_cnt])))
        #         self.conv_block_list.append(nn.Dropout2d(p=params['dropout_rate']))
        # SE block for feature enhancement (optional)
        if self.use_se_block:
            self.se_block = SE_Block(64)

        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)
        
        # self.resnet = resnet18(weights=None)
        # self.resnet.conv1 = nn.Conv2d(in_feat_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        # self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # Calculate the output shape of ResNet
        # with torch.no_grad():
        #     x = torch.zeros(1, *in_feat_shape[1:])

        #     if self.use_efficientnet:
        #         x = self.feature_extractor(x)
        #     else:
        #         for conv_cnt in range(len(self.conv_block_list)):
        #             x = self.conv_block_list[conv_cnt](x)
        #     breakpoint()        
        #     x = x.transpose(1, 2).contiguous()
        #     x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        #     # x = self.TCN(x)
        
        #     (x, _) = self.gru(x)
        #     resnet_out_shape = x.shape[1:] # 20, 512

        # Fully connected layer after ResNet
        # self.fc = nn.Linear(resnet_out_shape[0]  * resnet_out_shape[1], 512)
        
        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            torchaudio.models.Conformer(
                input_dim= params['rnn_size'] * 2,# resnet_out_shape[-1],
                num_heads=params['cf_num_heads'],  # 可以根据需要调整
                ffn_dim=params['cf_ffn_dim'],  # 可以根据需要调整
                num_layers=params['cf_num_layers'],  # 可以根据需要调整
                depthwise_conv_kernel_size=params['cf_depthwise_conv_kernel_size']
            )
            for _ in range(params['cf_num_cfs'])
        ])
        # Time pooling Output layer
        self.time_pool = nn.AdaptiveAvgPool1d(out_shape[1])
        self.output = nn.Linear(params['rnn_size'] * 2, out_shape[2])
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)
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
        # breakpoint()
        # Feature extraction
        
        # breakpoint()
        if self.use_efficientnet:
            x = self.feature_extractor(x)
            batch_size = x.size(0)  # Get batch size
            # Reshape (batch_size, 1280) -> (batch_size, 64, 20, 4)
            x = x.view(batch_size, 64, 20, 4)
        else:
            for conv_cnt in range(len(self.conv_block_list)):
                x = self.conv_block_list[conv_cnt](x)
        
        if self.use_se_block:
            x = self.se_block(x)  # Apply SE block only if enabled

        batch_size, channels, time, freq = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()


        (x, _) = self.gru(x)
        lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)
        
        for layer in self.conformer_layers: 
            x, _ = layer(x, lengths)   
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
        # breakpoint()
        return x






def main(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # ------------- Extract features and labels for development set -----------------------------
    # dev_feat_cls = cls_feature_class.FeatureClass(params)
    # # # # Extract labels
    # dev_feat_cls.generate_new_labels()  
    model = SELDConformer(in_feat_shape=(128, 7, 100, 128), out_shape= (128, 20, 156), params= params)
    
    # 构造随机输入，input shape应为 [batch_size, 1, time_steps]
    # 假设每个样本为1秒长，采样率为16000Hz
    input = torch.rand(128,7,100,128)  # batch_size为5, 1个通道, 16000个时间步
    print(model)
    # 运行模型
    output = model(input)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")




if __name__ == '__main__':
    # 初始化模型


    # 输出结果
    # x = torch.rand(128, 100, 45)  
    # avg_pool = nn.AvgPool1d(kernel_size=5, stride=5)  # 根据需要调整kernel_size和stride
    # x_pooled = avg_pool(x.transpose(1, 2)).transpose(1, 2)  # transpose是因为nn.AvgPool1d默认作用于最后一个维度
    # print(x_pooled.shape)
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
    