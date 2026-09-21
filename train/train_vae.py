import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.vae import VAE
from utils.data_utils import RNNDataset
from utils.logging_utils import (
    init_wandb,
    log_metrics,
    log_reconstructions,
    save_checkpoint,
)

import matplotlib
matplotlib.use("Agg")

CHECKPOINT_DIR = "./checkpoints/vae/"
WANDB_PROJECT_NAME = "world-models-vae"

def get_loaders(data_dir, batch_size=16, num_workers=0, shuffle=True, test_pct=0.2):
    ds = RNNDataset(data_dir)

    train_ds, test_ds = random_split(ds, (1-test_pct, test_pct))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_dl, test_dl


def dynamics_aware_loss(x_recon, x, lambda_dyn):
    "Weighted least squares approach to specifically identify the relevant dynamic pixels, and tell the model to focus more on them"
    background = torch.median(x, dim=1).values.unsqueeze(1).detach()
    
    dynamic = torch.abs(x - background)
    weight = 1 + lambda_dyn * dynamic
    
    error = (x_recon - x) ** 2
    
    # Normalize by the number of frames so that the loss scale does not depend
    # on sequence length (or batch size).
    loss = (weight * error).sum() / x.shape[1]
    
    # print(background.mean(), background.min(), background.max(), background.median(), background.std())
    # print(dynamic.min(), dynamic.max(), dynamic.mean(), dynamic.median(), dynamic.std())
    # print(weight.min(), weight.max(), weight.mean(), weight.median(), weight.std())
    # print(error.min(), error.max(), error.mean(), error.median(), error.std())
    # print(loss.min(), loss.max(), loss.mean(), loss.median(), loss.std())
    
    return loss
    

def vae_loss(x_recon, x, mu, logvar):
    # Compute reconstruction and KL losses per frame, then average over frames.
    recon_loss = F.mse_loss(x_recon, x, reduction="none")
    recon_loss = recon_loss.flatten(1).sum(dim=1).mean()

    # kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(dim=1).mean()

    return recon_loss + kld, recon_loss, kld


def train(data_dir, run_name, epochs=1, batch_size=16, num_workers=0, lr=1e-3,
          log_every=1000, num_eval_batches=10, beta_kl=1.0, lambda_dyn=1.0, latents_dim=64):
    if batch_size != 1:
        raise ValueError("Temporal VAE training currently supports batch_size=1 only")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device), log_every=log_every)
    
    init_wandb(config, WANDB_PROJECT_NAME, run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    print("Loading the model to device...")
    model = VAE(latents_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # use_amp = device.type == "cuda"
    # scaler = torch.GradScaler("cuda", enabled=use_amp)

    step_ckpt_dir = os.path.join(CHECKPOINT_DIR, "steps")
    epoch_ckpt_dir = os.path.join(CHECKPOINT_DIR, "epochs")

    global_step = 0
    running_loss = 0.0
    running_recon = 0.0
    running_kld = 0.0
    running_dyn_loss = 0.0
    running_count = 0
    
    print("Starting training")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kld = 0.0
        epoch_dyn_loss = 0.0

        print(f"Starting epoch {epoch}...")

        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for obs, acts in steps_bar:
            batch = obs.to(device)
            global_step += 1
            
            if batch.ndim < 5: raise AssertionError("Choose a Dataset that has temporal representation")
            B, T, C, H, W = batch.shape
            batch_flat = batch.reshape(B*T, C, H, W)
            
            recon_flat, mu, logvar = model(batch_flat.contiguous())
            recon = recon_flat.reshape(B, T, C, H, W)
            dyn_loss = dynamics_aware_loss(recon, batch, lambda_dyn)
            _, recon_loss, kld = vae_loss(recon_flat, batch_flat, mu, logvar)
            
            loss = recon_loss + beta * kld + lambda_dyn * dyn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_recon += recon_loss.item()
            running_kld += kld.item()
            running_dyn_loss += dyn_loss.item()
            running_count += batch.size(0)
            epoch_loss += loss.item()
            epoch_recon += recon_loss.item()
            epoch_kld += kld.item()
            epoch_dyn_loss += dyn_loss.item()

            # steps_bar.write(f"Finished training step {global_step}")
            if global_step % log_every == 0:
                avg_loss = running_loss / running_count
                avg_recon = running_recon / running_count
                avg_kld = running_kld / running_count
                avg_dyn_loss = running_dyn_loss / running_count

                # Subsampled evaluation for step-level checkpointing
                test_metrics = evaluate(
                    model,
                    test_dl,
                    device,
                    max_batches=num_eval_batches,
                    beta_kl=beta_kl,
                    lambda_dyn=lambda_dyn
                )

                log_metrics({
                    "train/loss": avg_loss,
                    "train/recon_loss": avg_recon,
                    "train/kld": avg_kld,
                    "train/dyn_loss": avg_dyn_loss,
                    "test/loss": test_metrics["loss"],
                    "test/recon_loss": test_metrics["recon_loss"],
                    "test/kld": test_metrics["kld"],
                    "test/dyn_loss": test_metrics["dyn_loss"],
                    "train/beta_kl": beta_kl,
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
                    model, optimizer,
                    save_dir=step_ckpt_dir,
                    filename=f"vae_step_{global_step:06d}.pt",
                    metadata={"global_step": global_step, "epoch": epoch,
                              "train_loss": avg_loss, "test_loss": test_metrics["loss"]},
                )

                steps_bar.write(f"Step {global_step} (epoch {epoch}) — train: {avg_loss:.4f}  test: {test_metrics['loss']:.4f}")

                running_loss = 0.0
                running_recon = 0.0
                running_kld = 0.0
                running_dyn_loss = 0.0
                running_count = 0

                model.train()

        # Epoch level checkpointing and evaluation (evaluation over the whole evaluation set)
        n_train = len(train_dl.dataset)
        avg_epoch_loss = epoch_loss / n_train
        avg_epoch_recon = epoch_recon / n_train
        avg_epoch_kld = epoch_kld / n_train
        avg_epoch_dyn_loss = epoch_dyn_loss / n_train

        print(f"\nEpoch {epoch} training done (step {global_step}) — "
              f"train_loss: {avg_epoch_loss:.4f}  recon: {avg_epoch_recon:.4f}  "
              f"kld: {avg_epoch_kld:.4f}  dyn: {avg_epoch_dyn_loss:.4f}, beta_kl: {beta_kl:.4f}")

        print("Running full evaluation on test set...")
        test_metrics = evaluate(model, test_dl, device, beta_kl=beta_kl)

        log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_recon_loss": avg_epoch_recon,
            "train/epoch_kld": avg_epoch_kld,
            "train/epoch_dyn_loss": avg_epoch_dyn_loss,
            "test/epoch_loss": test_metrics["loss"],
            "test/epoch_recon_loss": test_metrics["recon_loss"],
            "test/epoch_kld": test_metrics["kld"],
            "test/epoch_dyn_loss": test_metrics["dyn_loss"],
            "train/beta_kl": beta_kl,
            "epoch": epoch,
            "global_step": global_step,
        }, step=global_step)

        log_reconstructions(
            test_metrics["sample_originals"],
            test_metrics["sample_reconstructions"],
            step=global_step,
        )

        save_checkpoint(
            model, optimizer,
            save_dir=epoch_ckpt_dir,
            filename=f"vae_epoch_{epoch:03d}.pt",
            metadata={"epoch": epoch, "global_step": global_step,
                        "train_loss": avg_epoch_loss, "test_loss": test_metrics["loss"],
                        "beta_kl": beta_kl},
        )

        print(f"Epoch {epoch} complete — test_loss: {test_metrics['loss']:.4f}  "
              f"test_recon: {test_metrics['recon_loss']:.4f}  test_kld: {test_metrics['kld']:.4f}\n")

        model.train()


@torch.no_grad()
def evaluate(model, test_dl, device, max_batches=None, beta_kl=1.0, lambda_dyn=1.0):
    """Evaluation function. `max_batches` specifies the number of batches for evaluation, for when I need step-level checkpointing"""
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    total_dyn_loss = 0.0
    n_samples = 0
    sample_originals = None
    sample_reconstructions = None

    n_batches = min(max_batches, len(test_dl)) if max_batches else len(test_dl)
    eval_bar = tqdm(test_dl, desc="Evaluating", total=n_batches)

    for obs, acts in eval_bar:
        batch = obs.to(device)
        
        if batch.ndim < 5: raise AssertionError("Choose a Dataset that has temporal representation")
        B, T, C, H, W = batch.shape
        batch_flat = batch.reshape(B*T, C, H, W)
        
        recon_flat, mu, logvar = model(batch_flat.contiguous())
        recon = recon_flat.reshape(B, T, C, H, W)
        _, recon_loss, kld = vae_loss(recon_flat, batch_flat, mu, logvar)
        dyn_loss = dynamics_aware_loss(recon, batch, lambda_dyn)
        loss = recon_loss + beta_kl * kld + lambda_dyn * dyn_loss

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld.item()
        total_dyn_loss += dyn_loss.item()
        n_samples += batch.size(0)

        # Grab the first batch we see for visualization
        if sample_originals is None:
            sample_originals = batch[0].detach()
            sample_reconstructions = recon[0].detach()

        eval_bar.set_postfix(loss=f"{total_loss / n_samples:.4f}")

        if max_batches and eval_bar.n >= max_batches:
            break

    eval_bar.close()

    print(f"Evaluated on {n_samples} samples — "
          f"loss: {total_loss / n_samples:.4f}  "
          f"recon: {total_recon / n_samples:.4f}  "
            f"kld: {total_kld / n_samples:.4f}  "
            f"dyn: {total_dyn_loss / n_samples:.4f}")

    return {
        "loss": total_loss / n_samples,
        "recon_loss": total_recon / n_samples,
        "kld": total_kld / n_samples,
        "dyn_loss": total_dyn_loss / n_samples,
        "sample_originals": sample_originals,
        "sample_reconstructions": sample_reconstructions,
    }


if __name__ == "__main__":
    DATA_DIR = "./data/"
    
    run_name = input("Enter run name (Optional): ")
    
    train(DATA_DIR, run_name=run_name, batch_size=1, epochs=30, log_every=100, num_eval_batches=25, 
        beta_kl=2.0, lambda_dyn=3.0, latents_dim=128)
