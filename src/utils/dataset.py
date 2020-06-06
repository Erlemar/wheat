from typing import Tuple, Dict, Union

import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from omegaconf import DictConfig
import torch
import os
import cv2
from albumentations.core.composition import Compose


class WheatDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, cfg: DictConfig, transforms: Compose, mode: str = 'train', image_dir: str = ''
    ):
        """
        Prepare data for wheat competition.

        Args:
            dataframe: dataframe with image id and bboxes
            mode: train/val/test
            cfg: config with parameters
            image_dir: path to images
            transforms: albumentations
        """
        self.image_dir = image_dir
        self.df = dataframe
        self.mode = mode
        self.cfg = cfg
        self.image_ids = os.listdir(self.image_dir) if self.df is None else self.df['image_id'].unique()
        self.transforms = transforms

    def __getitem__(self, idx: int) -> Tuple[np.array, Dict[str, Union[torch.Tensor, np.array]], str]:
        image_id = self.image_ids[idx].split('.')[0]
        image = cv2.imread(f'{self.image_dir}/{image_id}.jpg', cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # normalization.
        # TO DO: refactor preprocessing
        image /= 255.0

        # test dataset must have some values so that transforms work.
        target = {
            'labels': torch.as_tensor([[0]], dtype=torch.float32),
            'boxes': torch.as_tensor([[0, 0, 0, 0]], dtype=torch.float32),
        }

        # for train and valid test create target dict.
        if self.mode != 'test':
            image_data = self.df.loc[self.df['image_id'] == image_id]
            boxes = image_data[['x', 'y', 'x1', 'y1']].values

            areas = image_data['area'].values
            areas = torch.as_tensor(areas, dtype=torch.float32)

            # there is only one class
            labels = torch.ones((image_data.shape[0],), dtype=torch.int64)
            iscrowd = torch.zeros((image_data.shape[0],), dtype=torch.int64)

            target['boxes'] = boxes
            target['labels'] = labels
            target['image_id'] = torch.tensor([idx])
            target['area'] = areas
            target['iscrowd'] = iscrowd

            if self.transforms:
                image_dict = {'image': image, 'bboxes': target['boxes'], 'labels': labels}
                image_dict = self.transforms(**image_dict)
                image = image_dict['image']
                target['boxes'] = torch.as_tensor(image_dict['bboxes'], dtype=torch.float32)

        else:
            image_dict = {'image': image, 'bboxes': target['boxes'], 'labels': target['labels']}
            image = self.transforms(**image_dict)['image']

        return image, target, image_id

    def __len__(self) -> int:
        return len(self.image_ids)
