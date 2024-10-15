import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x



class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder: Conv blocks with max pooling
        self.enc1 = ConvBlock(in_channels, 64)  # No change in resolution
        self.enc2 = ConvBlock(64, 128)  # Apply downsample by 2x2
        self.enc3 = ConvBlock(128, 256)  # Further downsample by 2x2

        # Adjust pool size to downsample in both time and mel dimensions as required
        self.pool1 = nn.MaxPool2d((3, 2))  # Downsample 1/3 time and 1/2 mel
        self.pool2 = nn.MaxPool2d((2, 2))  # Further downsample 1/2 time and 1/2 mel

        # Decoder: Conv blocks with upsampling
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def center_crop(self, enc, target_size):
        _, _, h, w = enc.size()
        target_h, target_w = target_size
        delta_h = h - target_h
        delta_w = w - target_w
        enc = enc[:, :, delta_h // 2: h - (delta_h - delta_h // 2), delta_w // 2: w - (delta_w - delta_w // 2)]
        return enc

    def forward(self, x):
        # Encoder
        # breakpoint()
        enc1 = self.enc1(x)  # No downsampling here  after 64, 100 ,128
        pol1 = self.pool1(enc1)
        enc2 = self.enc2(pol1)  # Downsampling by (3, 2) after unchanged 128, 33 ,64
        enc3 = self.enc3(self.pool2(enc2))  # Further downsampling by (2, 2) 256, 16, 32



        # Decoder
        dec2 = self.upconv2(enc3)    # 128, 32, 64
        # centor crop env2 to match dec2
        enc2 = self.center_crop(enc2, target_size=(dec2.size(2), dec2.size(3)))


        dec2 = torch.cat((dec2, enc2), dim=1)    # dec2.shape 128, 32, 64
        dec2 = self.dec2(dec2)   # dec2.shape 128, 32, 64

        dec1 = self.upconv1(dec2) # 
        dec1 = F.interpolate(dec1, size=(enc1.size(2), enc1.size(3)), mode='bilinear', align_corners=False)

        dec1 = torch.cat((dec1, enc1), dim=1) # enc1.shape torch.Size([1, 64, 100, 128]) dec1,shape [1, 64, 64, 128])
        dec1 = self.dec1(dec1)

        return self.final_conv(dec1)

class SELDUnet(torch.nn.Module):
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

        
        # U-Net for feature extraction
        self.unet = UNet(in_channels=in_feat_shape[1], out_channels=params['nb_cnn2d_filt'])

        # GRU for temporal modeling
        self.gru_input_dim = params['nb_cnn2d_filt'] * int(np.floor(in_feat_shape[-1] / np.prod(params['f_pool_size'])))
        self.gru = torch.nn.GRU(input_size=self.gru_input_dim, hidden_size=params['rnn_size'],
                                num_layers=params['nb_rnn_layers'], batch_first=True,
                                dropout=params['dropout_rate'], bidirectional=True)

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for mhsa_cnt in range(params['nb_self_attn_layers']):
            self.mhsa_block_list.append(nn.MultiheadAttention(embed_dim=self.params['rnn_size'], num_heads=self.params['nb_heads'], dropout=self.params['dropout_rate'], batch_first=True))
            self.layer_norm_list.append(nn.LayerNorm(self.params['rnn_size']))

        # Fusion layers
        if in_vid_feat_shape is not None:
            self.visual_embed_to_d_model = nn.Linear(in_features=int(in_vid_feat_shape[2] * in_vid_feat_shape[3]), out_features=self.params['rnn_size'])
            self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=self.params['rnn_size'], nhead=self.params['nb_heads'], batch_first=True)
            self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=self.params['nb_transformer_layers'])

        # Fully connected layers
        self.fnn_list = torch.nn.ModuleList()
        if params['nb_fnn_layers']:
            for fc_cnt in range(params['nb_fnn_layers']):
                self.fnn_list.append(nn.Linear(params['fnn_size'] if fc_cnt else self.params['rnn_size'], params['fnn_size'], bias=True))
        self.fnn_list.append(nn.Linear(params['fnn_size'] if params['nb_fnn_layers'] else self.params['rnn_size'], out_shape[-1], bias=True))

        self.doa_act = nn.Tanh()
        self.dist_act = nn.ReLU()

    def forward(self, x, vid_feat=None):
        """input: (batch_size, mic_channels, time_steps, mel_bins)"""
        # U-Net feature extraction
        breakpoint()
        x = self.unet(x)  # x.shape -> (batch_size, nb_cnn2d_filt, time_steps/5, mel_bins/4)
        for conv_cnt in range(1,len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)    

        # Reshape for GRU
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()

        # GRU
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]

        # Multi-head Self Attention
        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        # Vision features fusion (if applicable)
        if vid_feat is not None:
            vid_feat = vid_feat.view(vid_feat.shape[0], vid_feat.shape[1], -1)  # b x 50 x 49
            vid_feat = self.visual_embed_to_d_model(vid_feat)
            x = self.transformer_decoder(x, vid_feat)

        # Fully connected layers
        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)

        doa = self.fnn_list[-1](x)

        # Reshape for DOA and distance estimates
        doa = doa.reshape(doa.size(0), doa.size(1), 3, 4, 13)
        doa1 = doa[:, :, :, :3, :]
        dist = doa[:, :, :, 3:, :]

        doa1 = self.doa_act(doa1)
        dist = self.dist_act(dist)
        doa2 = torch.cat((doa1, dist), dim=3)

        doa2 = doa2.reshape((doa.size(0), doa.size(1), -1))
        return doa2
    


    if __name__ == '__main__':
        pass 