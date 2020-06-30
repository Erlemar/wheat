from omegaconf import DictConfig
from omegaconf import DictConfig
from torch import nn

from src.utils.utils import load_obj


class FasterRcnn(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.model = load_obj(cfg.model.backbone.class_name)
        self.model = self.model(**cfg.model.backbone.params)

        # get number of input features for the classifier
        self.output_dimension = self.model.roi_heads.box_predictor.cls_score.in_features

    def forward(self, x):
        pass
