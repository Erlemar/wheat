import os
import shutil
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.loggers import CometLogger, TensorBoardLogger
import numpy as np
from src.lightning_classes.lightning_wheat import LitWheat
from src.utils.utils import set_seed, format_prediction_string, collate_fn
from src.utils.get_dataset import get_test_dataset
from src.lightning_classes.lightning_wheat import LitWheat
import torch

def predict(cfg: DictConfig):
    """
    Run pytorch-lightning model

    Args:
        cfg: hydra config

    Returns:

    """
    set_seed(cfg.training.seed)

    test_dataset = get_test_dataset(cfg)

    model = LitWheat.load_from_checkpoint(
        r'd:\DataScience\Python_projects\Current_projects\wheat\outputs\2020_05_06_09_32_36\saved_models\_ckpt_epoch_0.ckpt')
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg.data.batch_size,
                                              num_workers=cfg.data.num_workers,
                                              shuffle=False,
                                              collate_fn=collate_fn)
    detection_threshold = 0.5
    results = []

    for images, _, image_ids in test_loader:

        images = list(image.to(cfg.general) for image in images)
        outputs = model(images)

        for i, image in enumerate(images):
            boxes = outputs[i]['boxes'].data.cpu().numpy()
            scores = outputs[i]['scores'].data.cpu().numpy()

            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            scores = scores[scores >= detection_threshold]
            image_id = image_ids[i]

            result = {
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            }

            results.append(result)

    model = LitWheat(hparams=cfg)

    early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)

    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)
    comet_logger = CometLogger(save_dir=cfg.general.save_dir,
                               workspace=cfg.general.workspace,
                               project_name=cfg.general.project_name,
                               # api_key=cfg.private.comet_api,
                               experiment_name=os.getcwd().split('\\')[-1])

    trainer = pl.Trainer(logger=[tb_logger, comet_logger],
                         early_stop_callback=early_stopping,
                         checkpoint_callback=model_checkpoint,
                         nb_sanity_val_steps=0,
                         gradient_clip_val=0.5,
                         **cfg.trainer)
    trainer.fit(model)


@hydra.main(config_path="conf/config.yaml")
def run_model(cfg: DictConfig) -> None:
    print(cfg.pretty())
    predict(cfg)


if __name__ == "__main__":
    run_model()
