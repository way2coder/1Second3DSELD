from .seldnet_model import *
from .conformer import *
from .seldnet_distance import * 
from .baseline_model import *
from .resnet_conformer import *
from .architecture.CST_former_model import *
from .SELDUnet import *
from .SCConv import *
# from .tcn_cst import *

models = {
    'SeldModel':SeldModel,
    'SELDConformer':SELDConformer,
    'SeldDistanceModel':SELDDistanceModule,
    'VanillaSeldModel':VanillaSeldModel,
    'ResNetConformer':ResNetConformer,
    'CST_former':CST_former,
    'CST_Conformer':CST_Conformer, 
    'SELDUnet' :SELDUnet,
    'SELDConformerEdit' : SELDConformerEdit,
    'GSELD':GSELD,
}