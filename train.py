import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from data import CIFAR10Data
from module import CIFAR10Module
from callbacks import *
from pathlib import Path
import wandb

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("name", None, "name used for wandb logger")
flags.DEFINE_string("init", None, "initial weights to use")
flags.DEFINE_integer("max_epochs", 1000, "number of epochs to run for")
flags.DEFINE_integer("precision", 32, "precision to use")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_integer("num_workers", 4, "number of workers to use for data loading")
flags.DEFINE_string("save", None, "output file to save model weights")
flags.DEFINE_bool("callbacks", True, "whether to compute gradient callbacks")
flags.DEFINE_bool(
    "fullbatch", False, "whether to aggregate batches to emulate full batch training"
)


def main(argv):
    seed_everything(FLAGS.seed)
    logger = WandbLogger(project="colt_final", name=FLAGS.name)
    logger.experiment.config.update(FLAGS)

    model = CIFAR10Module()
    if FLAGS.init is not None:
        model.load_state_dict(torch.load(Path(FLAGS.init)))
    data = CIFAR10Data()

    if FLAGS.callbacks:
        callbacks = [
            LearningRateMonitor(log_momentum=True),
            TimeEpoch(),
            TotalGradient(),
            WeightNorm(),
        ]
    else:
        callbacks = [LearningRateMonitor(log_momentum=True), TimeEpoch()]

    if FLAGS.fullbatch:
        accumulate_grad_batches = 50000 // FLAGS.batch_size
        log_every_n_steps = 1
    else:
        accumulate_grad_batches = 1
        log_every_n_steps = 50

    trainer = Trainer(
        logger=logger,
        gpus=1,
        max_epochs=FLAGS.max_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=50,
        log_every_n_steps=log_every_n_steps,
        precision=FLAGS.precision,
        deterministic=True,
        benchmark=True,
        terminate_on_nan=True,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer.fit(model, data)
    if FLAGS.save:
        save_file = Path(FLAGS.save)
        save_file.parent.mkdir(exist_ok=True, parents=True)
        torch.save(model.state_dict(), save_file)


if __name__ == "__main__":
    app.run(main)
