# The SELDnet architecture

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from IPython import embed
# from parameters import get_params
import sys
from torchlibrosa import STFT


# from seldnet_distance import SELDDistanceModule


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
        self.dist_linear = nn.Linear(self.nb_classes, self.nb_classes)
        self.dist_act = nn.ELU()
        self.distance_module = SELDDistanceModule(data_in=in_feat_shape, data_out=out_shape, nb_classes=self.nb_classes)

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_ bins)"""
        # x.shape 128, 7, 250, 64 vid_feat.shape 
        distance = self.distance_module(x)
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
        

        # breakpoint()
        for fnn_cnt in range(len(self.fnn_list) - 1): # linear(128, 128) linear(128, 156)
            x = self.fnn_list[fnn_cnt](x)
        doa = self.fnn_list[-1](x)

        doa = doa.reshape(doa.size(0), doa.size(1), 3, 4,  self.nb_classes) # b 50 3 4 13
        doa1 = doa[:, :, :, :3, :]  #[128, 50, 3, 3, 13]
        dist = doa[:, :, :, 3:, :]  #[128, 50, 3, 1, 13] 

        dist = dist * distance 
        dist = self.dist_linear(dist)


        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        
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
class SELDDistanceModule(nn.Module):
    def __init__(self, data_in, data_out,  nb_classes, kernels= 'square', n_grus = 2, att_conf = 'onAll'):
        super(SELDDistanceModule, self).__init__()
        self.n_fft = 512
        self.hop_length = 256
        self.nb_cnn2d_filt = 128
        self.pool_size = [8, 8, 2]
        self.rnn_size = [128, 128]
        self.fnn_size = int(data_out[-1]) // 4  # 3*4*nb*class 
        self._nb_classes = nb_classes
        self.kernels = kernels
        self.n_grus = n_grus
        # self.features_set = features_set
        self.att_conf = att_conf

        # kernels "freq" [1, 3] - "time" [3, 1] - "square" [3, 3]
        if self.kernels == "freq":
            self.kernels = (1,3)
        elif self.kernels == "time":
            self.kernels = (3,1)
        elif self.kernels == "square":
            self.kernels = (3,3)
        else:
            raise ValueError
        
        self.STFT = STFT(n_fft=self.n_fft, hop_length=self.hop_length)

        # feature set "stft", "sincos", "all"
        # if self.features_set == "stft":
        #     self.data_in = [1, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        # elif self.features_set == "sincos":
        #     self.data_in = [2, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        # elif self.features_set == "all":
        #     self.data_in = [3, (10*16000+self.n_fft)//(self.n_fft - self.hop_length) - 1,  int(self.n_fft/2)]
        # else: 
        #     raise ValueError

        self.data_in = data_in[1:]

        # ATTENTION MAP False, "onSpec", "onAll"
        if self.att_conf == "Nothing":
            pass
        elif self.att_conf == "onSpec":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = 1, kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )
        elif self.att_conf == "onAll":
            self.heatmap = nn.Sequential(
                nn.Conv2d(in_channels = self.data_in[0], out_channels = 16,
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(16),
                nn.ELU(),
                nn.Conv2d(in_channels = 16, out_channels = 64, 
                        kernel_size = (3,3), padding = "same", bias = False),
                nn.BatchNorm2d(64),
                nn.ELU(),
                nn.Conv2d(in_channels = 64, out_channels = self.data_in[0], kernel_size = 1, padding = "same"),
                nn.Sigmoid()
            )  
        else:
            raise ValueError         

        # First Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=self.data_in[0], out_channels=8, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_features=8)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, self.pool_size[0]))
        self.pool1avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[0]))
        
        # Second Convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=32, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, self.pool_size[1]))
        self.pool2avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[1]))

        # Third Convolutional layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=self.nb_cnn2d_filt, kernel_size=self.kernels, padding="same", bias = False)
        self.batch_norm3 = nn.BatchNorm2d(num_features=self.nb_cnn2d_filt)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, self.pool_size[2]))
        self.pool3avg = nn.AvgPool2d(kernel_size=(1, self.pool_size[2]))

        # GRUS 2, 1, 0
        if self.n_grus == 2:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[0], bidirectional=True, batch_first = True)
            self.gru2 = nn.GRU(input_size=self.rnn_size[0]*2, hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 1:
            self.gru1 = nn.GRU(input_size=int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), hidden_size=self.rnn_size[1], bidirectional=True, batch_first = True)
        elif self.n_grus == 0:
            self.gru_linear1 = nn.Linear(in_features = int(self.data_in[-1]* self.nb_cnn2d_filt / (self.pool_size[-3]*self.pool_size[-2]*self.pool_size[-1])), out_features = self.rnn_size[0])
            self.gru_linear2 = nn.Linear(in_features=self.rnn_size[0], out_features=self.rnn_size[1]*2)
        else:
            raise ValueError

        self.fc1 = nn.Linear(in_features=self.rnn_size[1]*2, out_features=self.fnn_size)
        # self.fc2 = nn.Linear(in_features=self.fnn_size, out_features = 1)

        self.final = nn.Linear(in_features = self.data_in[-2], out_features = 1)

    def normalize_tensor(self, x):
        mean = x.mean(dim = (2,3), keepdim = True)
        std = x.std(dim = (2,3), unbiased = False, keepdim = True)
        return torch.div((x - mean), std)

    def forward(self, x):
        # features extraction
        input = x 
        # x_real, x_imm = self.STFT(x)
        # # b, c, t, f = x_real.size()
        # magn = torch.sqrt(torch.pow(x_real, 2) + torch.pow(x_imm, 2))
        # magn = torch.log(magn**2 + 1e-7)
        # previous_magn = magn

        # angles_cos = torch.cos(torch.angle(x_real + 1j*x_imm))
        # angles_sin = torch.sin(torch.angle(x_real + 1j*x_imm))
        # magn = magn[:,:,:,:-1]
        # angles_cos = angles_cos[:,:,:,:-1]  # 128,1,63,256
        # angles_sin = angles_sin[:,:,:,:-1]  # 128,1,63,256  // 128, 7 
 
        # # set up feature set
        # if self.features_set == "stft":
        #     x = magn
        # elif self.features_set == "sincos":
        #     x = torch.cat((angles_cos, angles_sin), dim = 1)
        # elif self.features_set == "all":
        #     x = torch.cat((magn, angles_cos, angles_sin), dim = 1)  # 128, 3, 63, 256
        # else:
        #     raise ValueError
        
        x = self.normalize_tensor(input)

        # computation of the heatmap
        if self.att_conf == "Nothing":
            pass
        else:
            hm = self.heatmap(x)
            if self.att_conf == "onSpec":
                magn = magn * hm
                # x = torch.cat((magn, angles_cos, angles_sin), dim = 1) 
                x = self.normalize_tensor(x)
            elif self.att_conf == "onAll":
                x = x * hm


        # convolutional layers
        x = self.conv1(x)
        x = self.batch_norm1(x)  # 128,8, 128
        x = F.elu(x)
        x = self.pool1(x) + self.pool1avg(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pool2(x) + self.pool2avg(x) # 128,32,100,2 

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pool3(x) + self.pool3avg(x)

        # recurrent layers (if any)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        if self.n_grus == 2:
            x, _ = self.gru1(x)
            x, _ = self.gru2(x)
        elif self.n_grus == 1:
            x, _ = self.gru1(x)
        else:
            x = self.gru_linear1(x)
            x = self.gru_linear2(x)

        x = F.elu(self.fc1(x))  # 128, 100,128 
        avg_pool = nn.AvgPool1d(kernel_size=5, stride=5)  # 根据需要调整kernel_size和stride
        x_pooled = avg_pool(x.transpose(1, 2)).transpose(1, 2)  # transpose是因为nn.AvgPool1d默认作用于最后一个维度
        
        
        x = F.elu(x_pooled)  # 128, 100, 1 
        
        x = x.reshape(x.shape[0], x.shape[1], 3, 1 , self._nb_classes)

        return x 
 


import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  #  裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i   # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i-1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。
        
        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


if __name__ == '__main__':
    # 初始化模型
    model = SELDDistanceModule(data_in=(128, 7, 100, 128), data_out= (128, 20, 180), nb_classes = 15)
    
    # 构造随机输入，input shape应为 [batch_size, 1, time_steps]
    # 假设每个样本为1秒长，采样率为16000Hz
    input = torch.rand(128,7,100,128)  # batch_size为5, 1个通道, 16000个时间步

    # 运行模型
    output = model(input)

    # 输出结果
    # x = torch.rand(128, 100, 45)  
    # avg_pool = nn.AvgPool1d(kernel_size=5, stride=5)  # 根据需要调整kernel_size和stride
    # x_pooled = avg_pool(x.transpose(1, 2)).transpose(1, 2)  # transpose是因为nn.AvgPool1d默认作用于最后一个维度
    # print(x_pooled.shape)
    
