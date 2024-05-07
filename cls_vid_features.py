import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import cls_feature_class
import parameters
import sys

class VideoFeatures(nn.Module):
    def __init__(self):
        super(VideoFeatures, self).__init__()

        self.weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=self.weights)
        self.backbone = torch.nn.Sequential(*(list(self.model.children())[:-2]))
        self.backbone.eval()

        self.preprocess = self.weights.transforms()

    def forward(self, images):
        with torch.no_grad():

            preprocessed_images = [self.preprocess(image) for image in images]

            max_batch_size = 1000
            iter = (len(preprocessed_images) - 1) // max_batch_size + 1
            vid_features_part_list = []
            for i in range(iter):
                preprocessed_images_part = torch.stack(preprocessed_images[i * max_batch_size: (i + 1) * max_batch_size], dim = 0)
                vid_features_part = self.backbone(preprocessed_images_part)
                vid_features_part = torch.mean(vid_features_part, dim=1)
                vid_features_part_list.append(vid_features_part)
            vid_features = torch.cat(vid_features_part_list, dim = 0)    
            return vid_features


def main(argv):
    # Expects one input - task-id - corresponding to the configuration given in the parameter.py file.
    # Extracts features and labels relevant for the task-id
    # It is enough to compute the feature and labels once. 

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    # ------------- Extract features and labels for development set -----------------------------
    dev_feat_cls = cls_feature_class.FeatureClass(params)
    
    # Extract audio features and normalize them
    # dev_feat_cls.extract_all_feature()  
    dev_feat_cls.preprocess_features()

    # # Extract labels
    dev_feat_cls.extract_all_labels()

    # # Extract visual features
    if params['modality'] == 'audio_visual':
        dev_feat_cls.extract_visual_features()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
