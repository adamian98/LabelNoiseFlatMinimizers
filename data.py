import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from tqdm import trange
import torch
import numpy as np
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 256, "batch size")
flags.DEFINE_bool("DA", False, "whether to use data augmentation")
flags.DEFINE_bool(
    "rand_data", False, "whether to use random data/labels for adversarial init"
)


class CIFAR10Data(pl.LightningDataModule):
    def __init__(self):
        super().__init__()
        self.mean = torch.tensor([0.49139968, 0.48215841, 0.44653091])
        self.std = torch.tensor([0.24703223, 0.24348513, 0.26158784])

    def prepare_data(self):
        test_transform = T.Compose(
            [T.ToTensor(), T.Normalize(mean=self.mean, std=self.std)]
        )
        if FLAGS.DA:
            train_transform = T.Compose(
                [
                    T.RandomCrop(32, padding=4),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=self.mean, std=self.std),
                ]
            )
        else:
            train_transform = test_transform

        self.trainset = CIFAR10(
            "data", train=True, transform=train_transform, download=True
        )
        self.testset = CIFAR10(
            "data", train=False, transform=test_transform, download=True
        )

        if FLAGS.rand_data:
            rng = np.random.default_rng()
            data = np.repeat(self.trainset.data, 10, axis=0)
            targets = rng.integers(0, 10, (len(data),))
            for i in trange(len(data), desc="zeroing out pixels"):
                xidx, yidx = np.unravel_index(
                    rng.choice(32 * 32, 32 * 32 // 10), (32, 32)
                )
                data[i, xidx, yidx, :] = 0
            self.trainset.data, self.trainset.targets = data, targets

    def train_dataloader(self):
        dataloader = DataLoader(
            self.trainset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.testset,
            batch_size=FLAGS.batch_size,
            num_workers=FLAGS.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
