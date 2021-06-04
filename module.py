import torch
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from loss import LabelSmoothingLoss
from models import resnet18, vgg16
from bisect import bisect
from torch import nn
from absl import flags
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

FLAGS = flags.FLAGS

flags.DEFINE_enum("model", "resnet18", ["resnet18", "vgg16"], "model to use")
flags.DEFINE_enum(
    "norm_layer",
    "groupnorm",
    ["groupnorm", "batchnorm", "none"],
    "normalization layer to use for resnet",
)
flags.DEFINE_integer("group_size", 32, "channels per group")

flags.DEFINE_float("lr", 1, "learning rate")
flags.DEFINE_float("momentum", 0, "momentum")
flags.DEFINE_float("weight_decay", 0, "weight decay")
flags.DEFINE_float(
    "smoothing", 0, "probability of flipping a label for label smoothing/label noise"
)
flags.DEFINE_bool(
    "label_noise",
    False,
    "whether to use randomized label noise instead of label smoothing",
)
flags.DEFINE_float("warmup", 0, "length of warmup phase (between 0 and 1)")
flags.DEFINE_float(
    "div_start", float("inf"), "factor to divide learning rate by during warmup"
)
flags.DEFINE_float(
    "div_end", float("inf"), "factor to divide learning rate by during cosine annealing"
)
flags.DEFINE_bool("freezeBN", False, "whether to freeze batch norm during training")


class CIFAR10Module(pl.LightningModule):
    def __init__(self):
        super().__init__()
        model_dict = {
            "resnet18": resnet18,
            "vgg16": vgg16,
        }
        norm_dict = {
            "groupnorm": lambda x: nn.GroupNorm(x // FLAGS.group_size, x),
            "batchnorm": nn.BatchNorm2d,
            "none": nn.Identity,
        }
        self.criterion = LabelSmoothingLoss(
            10, FLAGS.smoothing, label_noise=FLAGS.label_noise
        )
        self.accuracy = Accuracy()
        self.model = model_dict[FLAGS.model](
            num_classes=10, norm_layer=norm_dict[FLAGS.norm_layer]
        )

    def forward(self, batch):
        if FLAGS.freezeBN:
            self.model.eval()
        x, y = batch
        output = self.model(x)
        loss, trueloss = self.criterion(output, y)
        _, predictions = output.max(-1)
        accuracy = 100 * predictions.eq(y).sum() / len(y)
        return loss, trueloss, accuracy

    def training_step(self, batch, batch_nb):
        loss, trueloss, accuracy = self.forward(batch)
        self.log("loss/train", trueloss, on_epoch=True)
        self.log("acc/train", accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        _, trueloss, accuracy = self.forward(batch)
        self.log("loss/val", trueloss)
        self.log("acc/val", accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,
            momentum=FLAGS.momentum,
        )
        if FLAGS.fullbatch:
            total_steps = FLAGS.max_epochs
        else:
            total_steps = FLAGS.max_epochs * len(self.train_dataloader())

        scheduler = {
            "scheduler": LinearWarmupCosineAnnealingLR(
                optimizer,
                FLAGS.warmup * total_steps,
                total_steps,
                FLAGS.lr / FLAGS.div_start,
                FLAGS.lr / FLAGS.div_end,
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]
