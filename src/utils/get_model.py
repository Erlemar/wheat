from typing import Any

from omegaconf import DictConfig

from src.utils.utils import load_obj


def get_wheat_model(cfg: DictConfig) -> Any:
    """
    Get model

    Args:
        cfg: config

    Returns:
        initialized model
    """
    model = load_obj(cfg.model.backbone.class_name)
    model = model(**cfg.model.backbone.params)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    head = load_obj(cfg.model.head.class_name)

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = head(in_features, cfg.model.head.params.num_classes)

    return model
