#!/usr/bin/env python3
"""
Frontier-ready DDP training script.

Usage (on Frontier inside an srun job):
    python ddp_main.py <total_epochs> <save_every> --batch_size 32

Important:
- Launch with srun (or via sbatch job that calls srun):
    srun -n <world_size> python ddp_main.py 3 1 --batch_size 32
"""

import os
import socket
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datautils import MyTrainDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import time

# -----------------------
# DDP / SLURM helpers
# -----------------------
def ddp_setup():
    """Read SLURM-provided environment variables and initialize the process group."""
    rank = int(os.environ.get("SLURM_PROCID", os.environ.get("RANK", "0")))
    world_size = int(os.environ.get("SLURM_NTASKS", os.environ.get("WORLD_SIZE", "1")))
    local_rank = int(os.environ.get("SLURM_LOCALID", os.environ.get("LOCAL_RANK", "0")))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, local_rank, world_size

def ddp_cleanup():
    try:
        dist.barrier()
    except Exception:
        pass
    try:
        dist.destroy_process_group()
    except Exception:
        pass

# -----------------------
# Trainer class
# -----------------------
class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer,
                 gpu_id: int, save_every: int):
        """
        gpu_id: local device index on the node (LOCAL_RANK)
        """
        self.gpu_id = gpu_id
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        if torch.cuda.is_available():
            self.model = DDP(model, device_ids=[gpu_id])
        else:
            self.model = DDP(model)

        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        # batch size of this DataLoader (may vary if last smaller)
        try:
            b_sz = len(next(iter(self.train_data))[0])
        except StopIteration:
            b_sz = 0

        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}", flush=True)

        # If using DistributedSampler, set epoch for shuffling
        sampler = getattr(self.train_data, "sampler", None)
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        epoch_loss = 0.0
        steps = 0
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            epoch_loss += loss
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        print(f"[GPU{self.gpu_id}] Epoch {epoch} finished. Avg loss: {avg_loss:.6f}", flush=True)
        return avg_loss

    def _save_checkpoint(self, epoch, out_path_prefix="checkpoint"):
        """
        Save the model checkpoint in your Frontier project folder.
        Only the underlying module's state_dict is saved.
        """
        CHECKPOINT_DIR = "/lustre/orion/csc662/world-shared/topcicekd/projects/ddp_frontier_test"
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        PATH = os.path.join(CHECKPOINT_DIR, f"{out_path_prefix}_epoch{epoch}.pt")
        torch.save(self.model.module.state_dict(), PATH)

        print(f"[INFO][GPU{self.gpu_id}] Checkpoint saved at epoch {epoch}", flush=True)

    def train(self, max_epochs: int, rank: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if rank == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

# -----------------------
# Data / model helpers
# -----------------------
def load_train_objs():
    # Larger dataset for longer runtime
    train_set = MyTrainDataset(50000)

    # Bigger model for more compute
    model = torch.nn.Sequential(
        torch.nn.Linear(20, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 2)
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    sampler = DistributedSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=False
    )

# -----------------------
# main
# -----------------------
def main(total_epochs: int, save_every: int, batch_size: int):
    rank, local_rank, world_size = ddp_setup()
    slurm_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    num_workers = max(0, slurm_cpus - 1)

    dataset, model, optimizer = load_train_objs()
    train_data = prepare_dataloader(dataset, batch_size, num_workers)

    trainer = Trainer(model, train_data, optimizer, local_rank, save_every)

    try:
        if rank == 0:
            print(f"[INFO] Starting training on host {socket.gethostname()} | "
                  f"world_size={world_size} | rank={rank} | local_rank={local_rank}", flush=True)
        trainer.train(total_epochs, rank)
    except Exception as e:
        print(f"[ERROR] Exception on rank {rank}: {e}", flush=True)
        raise
    finally:
        ddp_cleanup()

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Frontier-ready DDP training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot (epochs)')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size per GPU')
    args = parser.parse_args()

    main(args.total_epochs, args.save_every, args.batch_size)
