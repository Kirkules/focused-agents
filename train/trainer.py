"""
Training loop for the GPT language model.

Implements:
  - AdamW with separate weight decay groups (decay matrices, not biases/norms)
  - Linear warmup + cosine decay LR schedule
  - Gradient norm clipping
  - Periodic validation loss evaluation
  - Checkpoint save/load
"""

import math
import os
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader

from model.model import GPT
from train.dataset import TokenDataset


@dataclass
class TrainerConfig:
    max_steps: int = 10_000
    batch_size: int = 32
    warmup_steps: int = 500
    max_lr: float = 3e-4
    min_lr: float = 3e-5
    grad_clip: float = 1.0
    weight_decay: float = 0.1
    eval_interval: int = 500
    checkpoint_dir: str | None = None


class Trainer:
    """
    Manages one training run: optimizer, LR schedule, gradient clipping,
    periodic evaluation, and checkpointing.

    Args:
        model: the GPT model to train.
        config: hyperparameters for the training run.
        train_path: path to the binary token file used for training.
        val_path: path to the binary token file used for validation.
    """

    def __init__(
        self,
        model: GPT,
        config: TrainerConfig,
        train_path: str,
        val_path: str,
    ) -> None:
        self.model = model
        self.config = config

        context_length = model.config.context_length
        self.train_loader = DataLoader(
            TokenDataset(train_path, context_length),
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            TokenDataset(val_path, context_length),
            batch_size=config.batch_size,
            shuffle=False,
            drop_last=True,
        )

        self.optimizer = self._build_optimizer()

        if config.checkpoint_dir is not None:
            os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.AdamW:
        """
        Build AdamW with two parameter groups:
          - decay: 2D+ weight matrices (linear layers, etc.)
          - no_decay: 1D params (LayerNorm scale/shift, biases, embeddings)

        Embedding tables are 2D in shape but should not be weight-decayed
        because they are lookup tables, not weight matrices in the traditional
        sense — decaying them shrinks infrequent token representations toward
        zero without any gradient signal to counteract it.
        """
        decay_params = []
        nodecay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Exclude embeddings from decay even though they are 2D
            if param.dim() >= 2 and "emb" not in name:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.config.max_lr)

    def get_lr(self, step: int) -> float:
        """
        Linear warmup from 0 to max_lr over warmup_steps, then cosine decay
        from max_lr to min_lr over the remaining steps.

        Args:
            step: current optimizer step (0-indexed).

        Returns:
            Learning rate to use at this step.
        """
        if step <= self.config.warmup_steps:
            # Linear warmup: step 0 → 0, step warmup_steps → max_lr
            return self.config.max_lr * step / self.config.warmup_steps

        # Cosine decay from warmup end to max_steps
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        # progress ∈ [0, 1]; at 1 the cosine term is -1, giving min_lr
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + cosine * (self.config.max_lr - self.config.min_lr)

    def _set_lr(self, step: int) -> None:
        lr = self.get_lr(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def _clip_and_step(self) -> None:
        """Clip gradient norm then apply the optimizer step."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        self.optimizer.step()
        self.optimizer.zero_grad()

    def evaluate(self) -> float:
        """
        Compute mean validation loss over the full val set without gradients.

        Returns:
            Mean cross-entropy loss (float).
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                _, loss = self.model(x, targets=y)
                total_loss += loss.item()
                n_batches += 1

        self.model.train()
        return total_loss / max(1, n_batches)

    def save_checkpoint(self, step: int) -> None:
        """
        Save model weights, optimizer state, and step count to a checkpoint.

        Args:
            step: the optimizer step this checkpoint corresponds to.
        """
        if self.config.checkpoint_dir is None:
            return
        path = os.path.join(self.config.checkpoint_dir, f"step_{step}.pt")
        torch.save(
            {
                "step": step,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def train(self) -> list[float]:
        """
        Run training for max_steps steps.

        Returns:
            List of per-step training losses.
        """
        self.model.train()
        self.optimizer.zero_grad()

        losses = []
        step = 0
        train_iter = iter(self.train_loader)

        while step < self.config.max_steps:
            # Exhaust and restart the iterator as needed
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                x, y = next(train_iter)

            self._set_lr(step)
            _, loss = self.model(x, targets=y)
            loss.backward()
            self._clip_and_step()

            losses.append(loss.item())
            step += 1

            if step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"step {step}: train={losses[-1]:.4f}  val={val_loss:.4f}")

            if self.config.checkpoint_dir and step % self.config.eval_interval == 0:
                self.save_checkpoint(step)

        return losses
