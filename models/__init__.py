from .seldnet_model import *
from .conformer import *
from .seldnet_distance import * 
models = {
    'SeldModel':SeldModel,
    'SeldConModel':SeldConModel,
    'SeldDistanceModel':SELDDistanceModule,
}