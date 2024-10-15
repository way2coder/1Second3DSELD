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
from torchvision.models import resnet18
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class VanillaSeldModel(torch.nn.Module):
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

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
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

class ResNetConformer(nn.Module):
    def __init__(self, in_feat_shape, out_shape, params, in_vid_feat_shape=None):
        super().__init__()
        self.nb_classes = params['unique_classes']
        self.params = params
        
        # ResNet feature extractor
        self.resnet = resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(in_feat_shape[1], 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])  # Remove avgpool and fc layers
        
        # breakpoint()
        # Calculate the output shape of ResNet
        with torch.no_grad():
            dummy_input = torch.zeros(1, *in_feat_shape[1:])
            resnet_out = self.resnet(dummy_input)
            resnet_out_shape = resnet_out.shape[1:]

        # Fully connected layer after ResNet
        self.fc = nn.Linear(resnet_out_shape[0]  * resnet_out_shape[2], 512)
        
        # Conformer layers
        self.conformer_layers = nn.ModuleList([
            torchaudio.models.Conformer(
                input_dim= resnet_out_shape[0],
                num_heads=4,  # 可以根据需要调整
                ffn_dim=1024,  # 可以根据需要调整
                num_layers=2,  # 可以根据需要调整
                depthwise_conv_kernel_size=31
            )
            for _ in range(8)
        ])
        
        # Time pooling
        self.time_pool = nn.AdaptiveAvgPool1d(out_shape[1])
        
        # Output layer
        self.output = nn.Linear(512, out_shape[2])
        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()


    def forward(self, x):
        # ResNet feature extraction
        # x.shape = in_feat_shape=(128, 7, 100, 128)
        x = self.resnet(x)  # x.shape = ([128, 512, 4, 4])
        batch_size, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, time, -1) # 128, 4, 2048
        
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        
        # Fully connected layer 
        x = self.fc(x)  
        # Conformer layers   
        lengths = torch.full((batch_size,), time, dtype=torch.long)  # 所有序列长度都是 4

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
        return x






def main(argv):
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # ------------- Extract features and labels for development set -----------------------------
    # dev_feat_cls = cls_feature_class.FeatureClass(params)
    # # # # Extract labels
    # dev_feat_cls.generate_new_labels()  
    model = VanillaSeldModel(in_feat_shape=(128, 7, 100, 128), out_shape= (128, 20, 156), params= params)
    
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

    