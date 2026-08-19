import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split

from models.vae import VAE
from utils.data_utils import MsPacmanDataset
from utils.logging_utils import init_wandb, log_metrics, log_reconstructions, save_checkpoint

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


def train(data_dir, run_name, epochs=1, batch_size=16, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device))
    
    init_wandb(config, "world-models-vae", run_name)

    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size)

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kld = 0.0

        for batch in tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}"):
            batch = batch.to(device)

            recon, mu, logvar = model(batch)
            loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kld += kld.item()

        n_train = len(train_dl.dataset)
        avg_train_loss = train_loss / n_train
        avg_train_recon = train_recon / n_train
        avg_train_kld = train_kld / n_train

        test_metrics = evaluate(model, test_dl, device)

        # Logging
        log_metrics({
            "train/loss": avg_train_loss,
            "train/recon_loss": avg_train_recon,
            "train/kld": avg_train_kld,
            "test/loss": test_metrics["loss"],
            "test/recon_loss": test_metrics["recon_loss"],
            "test/kld": test_metrics["kld"],
            "epoch": epoch,
        }, step=epoch)

        log_reconstructions(
            test_metrics["sample_originals"], 
            test_metrics["sample_reconstructions"], 
            step=epoch,
        )

        # Checkpointing
        save_checkpoint(model, optimizer, epoch, avg_train_loss, test_metrics["loss"], CHECKPOINT_DIR)

        print(f"Epoch {epoch} — train: {avg_train_loss:.4f}  test: {test_metrics['loss']:.4f}")


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
    
    train(DATA_DIR, run_name="vae-train-001")