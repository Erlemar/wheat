from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils.coco_eval import CocoEvaluator
from src.utils.coco_utils import get_coco_api_from_dataset, _get_iou_types
from src.utils.get_dataset import get_training_datasets
from src.utils.get_model import get_wheat_model
from src.utils.utils import load_obj, collate_fn


class LitWheat(pl.LightningModule):
    def __init__(self, hparams: Dict[str, float], cfg: DictConfig):
        super(LitWheat, self).__init__()
        self.cfg = cfg
        self.hparams: Dict[str, float] = hparams
        self.model = get_wheat_model(self.cfg)
        # if hasattr(self.model, 'parameters'):
        #     self.hparams['n_params'] = sum(p.numel() for p in self.model.parameters())

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def prepare_data(self):
        datasets = get_training_datasets(self.cfg)
        self.train_dataset = datasets['train']
        self.valid_dataset = datasets['valid']

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=True,
            collate_fn=collate_fn,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # prepare coco evaluator
        self.coco = get_coco_api_from_dataset(valid_loader.dataset)
        self.iou_types = _get_iou_types(self.model)
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)

        return valid_loader

    def configure_optimizers(self):
        if 'decoder_lr' in self.cfg.optimizer.params.keys():
            params = [
                {'params': self.model.decoder.parameters(), 'lr': self.cfg.optimizer.params.lr},
                {'params': self.model.encoder.parameters(), 'lr': self.cfg.optimizer.params.decoder_lr},
            ]
            optimizer = load_obj(self.cfg.optimizer.class_name)(params)

        else:
            optimizer = load_obj(self.cfg.optimizer.class_name)(self.model.parameters(), **self.cfg.optimizer.params)
        scheduler = load_obj(self.cfg.scheduler.class_name)(optimizer, **self.cfg.scheduler.params)

        return [optimizer], [{'scheduler': scheduler,
                              'interval': self.cfg.scheduler.step,
                              'monitor': self.cfg.scheduler.monitor}]

    def training_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        # separate losses
        loss_dict = self.model(images, targets)
        # total loss
        losses = sum(loss for loss in loss_dict.values())

        return {'loss': losses, 'log': loss_dict, 'progress_bar': loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets, image_ids = batch
        targets = [{k: v for k, v in t.items()} for t in targets]
        outputs = self.model(images, targets)
        res = {target['image_id'].item(): output for target, output in zip(targets, outputs)}
        self.coco_evaluator.update(res)

        return {}

    def validation_epoch_end(self, outputs):
        self.coco_evaluator.synchronize_between_processes()
        self.coco_evaluator.accumulate()
        self.coco_evaluator.summarize()
        # coco main metric
        metric = self.coco_evaluator.coco_eval['bbox'].stats[0]
        metric = torch.as_tensor(metric)
        tensorboard_logs = {'main_score': metric}
        # re-initialise coco_evaluator after end of each epoch 
        self.coco_evaluator = CocoEvaluator(self.coco, self.iou_types)
        return {'val_loss': metric, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}
