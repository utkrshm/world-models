import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.vae import VAE
from utils.data_utils import MsPacmanDataset
from utils.logging_utils import (
    init_wandb,
    log_metrics,
    log_reconstructions,
    save_checkpoint,
)

CHECKPOINT_DIR = "./checkpoints/vae/"

def get_loaders(data_dir, batch_size=16, shuffle=True, test_pct=0.2):
    ds = MsPacmanDataset(data_dir)

    train_ds, test_ds = random_split(ds, (1-test_pct, test_pct))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)
    
    return train_dl, test_dl


def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.mse_loss(x_recon, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld


def train(data_dir, run_name, epochs=1, batch_size=16, lr=1e-3, log_every=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device), log_every=log_every)
    
    init_wandb(config, "world-models-vae", run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size)

    print("Loading the model to device...")
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_ckpt_dir = os.path.join(CHECKPOINT_DIR, "steps")
    epoch_ckpt_dir = os.path.join(CHECKPOINT_DIR, "epochs")

    global_step = 0
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_count = 0

    print("Starting training")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0

        print(f"Starting epoch {epoch}...")

        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for batch in steps_bar:
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            running_count += batch.size(0)
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()

            steps_bar.write(f"Finished training step {global_step}")
            if global_step % log_every == 0:
                avg_loss = running_loss / running_count
                avg_recon = running_recon / running_count
                avg_kld = running_kld / running_count

                test_metrics = evaluate(model, test_dl, device)

                log_metrics({
                    "train/loss": avg_loss,
                    "train/recon_loss": avg_recon,
                    "train/kld": avg_kld,
                    "test/loss": test_metrics["loss"],
                    "test/recon_loss": test_metrics["recon_loss"],
                    "test/kld": test_metrics["kld"],
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)

                log_reconstructions(
                    test_metrics["sample_originals"], 
                    test_metrics["sample_reconstructions"], 
                    step=global_step,
                )

                # Step-wise checkpointing
                save_checkpoint(
                    model.cpu(), optimizer,				# Really important here to change the model to cpu, otherwise it'll choke
                    save_dir=step_ckpt_dir,
                    filename=f"vae_step_{global_step:06d}.pt",
                    metadata={"global_step": global_step, "epoch": epoch,
                              "train_loss": avg_loss, "test_loss": test_metrics["loss"]},
                )

                steps_bar.write(f"Step {global_step} (epoch {epoch}) — train: {avg_loss:.4f}  test: {test_metrics['loss']:.4f}")

                running_loss = 0.0
                running_recon = 0.0
                running_kld = 0.0
                running_count = 0

                model.train()

        # Epoch-wise checkpointing
        n_train = len(train_dl.dataset)
        avg_epoch_loss = epoch_loss / n_train
        avg_epoch_recon = epoch_recon / n_train
        avg_epoch_kld = epoch_kld / n_train

        save_checkpoint(
            model, optimizer,
            save_dir=epoch_ckpt_dir,
            filename=f"vae_epoch_{epoch:03d}.pt",
            metadata={"epoch": epoch, "global_step": global_step,
                      "train_loss": avg_epoch_loss},
        )

        print(f"Epoch {epoch} complete (step {global_step}) — "
              f"train_loss: {avg_epoch_loss:.4f}  recon: {avg_epoch_recon:.4f}  kld: {avg_epoch_kld:.4f}")


@torch.no_grad()
def evaluate(model, test_dl, device):
    """Run a full evaluation pass over the test set. Returns a metrics dict."""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    sample_originals = None
    sample_reconstructions = None

    for batch in test_dl:
        batch = batch.to(device)

        recon, mu, logvar = model(batch)
        loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()

        # Grab the first batch we see for visualization
        if sample_originals is None:
            sample_originals = batch.detach()
            sample_reconstructions = recon.detach()

    n_test = len(test_dl.dataset)

    return {
        "loss": total_loss / n_test,
        "recon_loss": total_recon / n_test,
        "kld": total_kld / n_test,
        "sample_originals": sample_originals,
        "sample_reconstructions": sample_reconstructions,
    }


if __name__ == "__main__":
    DATA_DIR = "./data/"
    
    run_name = input("Enter run name (Optional): ")
    
    train(DATA_DIR, run_name=run_name, batch_size=64, log_every=10)
