import torch
from pytorch_lightning.callbacks.base import Callback
from torchvision import transforms as T
from time import perf_counter
from absl import flags

FLAGS = flags.FLAGS


class WeightNorm(Callback):
    def on_train_epoch_end(self, trainer, pl_module, outputs):
        weightnorm = sum(
            param.square().sum() for param in pl_module.parameters()
        ).sqrt()
        pl_module.log("weight/norm", weightnorm)


class TotalGradient(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.mean_grad = [torch.zeros_like(param) for param in pl_module.parameters()]
        self.grad_sq_sum = 0
        if FLAGS.label_noise:
            self.noisy_sq_sum = 0
        self.num_steps = 0

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if not FLAGS.fullbatch:
            if FLAGS.label_noise:
                noisy_grads = [param.grad for param in pl_module.parameters()]
                noisy_sq = sum(grad.square().sum() for grad in noisy_grads)
                self.noisy_sq_sum += noisy_sq

                x, y = batch
                x, y = x.cuda(), y.cuda()
                _, true_loss, _ = pl_module.forward((x, y))
                grads = torch.autograd.grad(true_loss, pl_module.parameters())
                grad_sq = sum(grad.square().sum() for grad in grads)

                pl_module.log("grad/noisy_norm(batch)", noisy_sq.sqrt())
                pl_module.log("grad/norm(batch)", grad_sq.sqrt())
            else:
                grads = [param.grad for param in pl_module.parameters()]
                grad_sq = sum(grad.square().sum() for grad in grads)
                pl_module.log("grad/norm(batch)", grad_sq.sqrt())

            self.grad_sq_sum += grad_sq
            self.mean_grad = [
                grad1 + grad2 for grad1, grad2 in zip(self.mean_grad, grads)
            ]
            self.num_steps += 1

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if FLAGS.fullbatch:
            grad_sq = sum(param.grad.square().sum() for param in pl_module.parameters())
            pl_module.log("grad/norm", grad_sq.sqrt())
        else:
            self.mean_grad = [grad / self.num_steps for grad in self.mean_grad]
            mean_grad_sq = sum(grad.square().sum() for grad in self.mean_grad)
            self.grad_sq_sum /= self.num_steps
            pl_module.log("grad/norm", mean_grad_sq.sqrt())
            pl_module.log("grad/reg", self.grad_sq_sum - mean_grad_sq)
            if FLAGS.label_noise:
                self.noisy_sq_sum /= self.num_steps
                pl_module.log(
                    "grad/tr_G",
                    FLAGS.batch_size * (self.noisy_sq_sum - self.grad_sq_sum),
                )


class TimeEpoch(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        self.t = perf_counter()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        elapsed_t = perf_counter() - self.t
        pl_module.log("time/sec_per_epoch", elapsed_t)
        self.t = 0
