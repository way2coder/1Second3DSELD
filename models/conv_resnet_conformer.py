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
                ffn_dim=512,  # 可以根据需要调整
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
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # ResNet feature extraction
        # x.shape = in_feat_shape=(128, 7, 100, 128)
        x = self.resnet(x)  # x.shape = ([128, 512, 4, 4])
        batch_size, channels, time, freq = x.shape
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, time, -1) # 128, 4, 2048
        
        # Fully connected layer 
        x = self.fc(x)  
        # Conformer layers   
        lengths = torch.full((batch_size,), time, dtype=torch.long, device = x.device)  # 所有序列长度都是 4

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






# def main(argv):
#     task_id = '1' if len(argv) < 2 else argv[1]
#     params = parameters.get_params(task_id)

#     # ------------- Extract features and labels for development set -----------------------------
#     # dev_feat_cls = cls_feature_class.FeatureClass(params)
#     # # # # Extract labels
#     # dev_feat_cls.generate_new_labels()  
#     model = VanillaSeldModel(in_feat_shape=(128, 7, 100, 128), out_shape= (128, 20, 156), params= params)
    
#     # 构造随机输入，input shape应为 [batch_size, 1, time_steps]
#     # 假设每个样本为1秒长，采样率为16000Hz
#     input = torch.rand(128,7,100,128)  # batch_size为5, 1个通道, 16000个时间步
#     print(model)
#     # 运行模型
#     output = model(input)
#     print(output.shape)
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total number of parameters: {total_params}")




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
    