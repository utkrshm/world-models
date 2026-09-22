import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from models.vae import VAE
from models.mdrnn import Memory
from utils.data_utils import RNNDataset
from utils.logging_utils import (
    init_wandb,
    log_metrics,
    log_predictions,
    save_checkpoint, load_checkpoint,
)

CHECKPOINT_DIR = "./checkpoints/rnn/"
WANDB_PROJECT_NAME = "world-models-rnn"


def get_loaders(data_dir, batch_size=16, num_workers=2, shuffle=True, test_pct=0.2):
    
    def collate_fn(batch):
        # batch = [(obs[0], acts[0], lives[0]), ..., (obs[i], acts[i], lives[i])]
        observations = [ep[0] for ep in batch]
        actions = [ep[1] for ep in batch]
        lives = [ep[2] for ep in batch]
        
        lengths = torch.tensor([len(obs) for obs in observations])
        
        observations = pad_sequence(observations, batch_first=True)
        actions = pad_sequence(actions, batch_first=True)
        lives = pad_sequence(lives, batch_first=True)
        
        # Return padded sequence, along with original sequnence length
        return observations, actions, lives, lengths

    ds = RNNDataset(data_dir)

    train_ds, test_ds = random_split(ds, (1-test_pct, test_pct))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_dl, test_dl


def load_vae_frozen(weights_path, device, latents_dim):
    if isinstance(weights_path, str):
        weights_path = Path(weights_path)
    
    assert weights_path.exists() and weights_path.is_file(), "The provided path to the VAE's weights file does not exist"
    assert weights_path.suffix == ".pt", "VAE weights file is not a .pt file, please ensure that you load correct weights"

    vae = VAE(hidden_size=latents_dim).to(device=device)
    checkpoint = torch.load(weights_path, weights_only=True)
    vae.load_state_dict(checkpoint["model_state_dict"])
    
    for params in vae.parameters():
        params.requires_grad_(False)
    
    vae.eval()
    
    return vae

# Get the log PDF of a Gaussian distribution for each latent dimension.
def log_gauss_dist(y, mus, sigmas):
    sigmas = sigmas.float().clamp_min(1e-5)
    normalized = (y.float() - mus.float()) / sigmas
    return -torch.log(sigmas) - 0.5 * np.log(2 * np.pi) - 0.5 * normalized.square()

def nll_loss(weights, mus, sigmas, true_latents):
    true_latents = true_latents.unsqueeze(2)                # (b, seq_len, 1, z_size)

    log_density = log_gauss_dist(true_latents, mus, sigmas)
    log_joint_density = log_density.sum(dim=3)
    log_weights = torch.log(weights.float().clamp_min(1e-8))

    return -torch.logsumexp(log_weights + log_joint_density, dim=2)

def get_masks(lengths):
    max_len = max(lengths)
    
    positions = torch.arange(max_len).unsqueeze(0).to(device=lengths.device)      # (1, max_len)
    lengths_col = lengths.unsqueeze(1)                         # (b, 1)
    mask = (positions < lengths_col).int()                      # (b, max_len)
    return mask

# We get weights, mus, sigmas, hiddens from the model; lengths from the DataLoader; true_latents as the ground truth obs
def rnn_loss(weights, mus, sigmas, hiddens, lengths, true_latents):
    lengths_masks = lengths - 1          # To account for the shifting for the true latents
    loss = nll_loss(weights, mus, sigmas, true_latents)
    masks = get_masks(lengths_masks)

    # Mask loss to exclude contributions from ahead-of-time calculations
    masked_loss = loss * masks
    total_loss = masked_loss.sum()
    real_pos = masks.sum()
    
    return total_loss / real_pos    


def lives_loss(lives_logits, true_lives, lengths):
    """Compute masked cross-entropy for the lives after each transition."""
    transition_lengths = (lengths - 1).clamp_min(0)
    masks = get_masks(transition_lengths).bool()

    per_step_loss = F.cross_entropy(
        lives_logits.transpose(1, 2),
        true_lives,
        reduction="none",
    )
    return (per_step_loss * masks).sum() / masks.sum().clamp_min(1)


def build_prediction_samples(vae, observations, actions, weights, mus, lengths, num_seqs=8, max_frames=16):
    take = min(num_seqs, observations.size(0))
    max_transitions = min(max_frames, weights.size(1))
    frame_shape = observations.shape[2:]

    observations = observations[:take, : max_transitions + 1]
    actions = actions[:take, :max_transitions]
    weights = weights[:take, :max_transitions]
    mus = mus[:take, :max_transitions]

    transition_lengths = (lengths[:take] - 1).clamp(min=0, max=max_transitions)
    valid = torch.arange(max_transitions, device=observations.device).unsqueeze(0)
    valid = valid < transition_lengths.unsqueeze(1)

    pred_latents = (weights.float().unsqueeze(3) * mus.float()).sum(dim=2)
    pred = vae.decode(pred_latents.flatten(0, 1))
    pred = pred.view(take, max_transitions, *frame_shape)

    return {
        "org": observations[:, :-1][valid],
        "pred": pred[valid],
        "gt": observations[:, 1:][valid],
        "act": actions[valid],
    }


def train(data_dir, vae_weights_path, run_name, latents_dim, epochs=1, batch_size=16, num_workers=0, lr=1e-3, log_every=1000, num_eval_batches=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device), log_every=log_every)
    
    init_wandb(config, WANDB_PROJECT_NAME, run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    print("Loading the model to device...")
    vae = load_vae_frozen(vae_weights_path, device, latents_dim)
    model = Memory(latents_dim=latents_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    step_ckpt_dir = os.path.join(CHECKPOINT_DIR, "steps")
    epoch_ckpt_dir = os.path.join(CHECKPOINT_DIR, "epochs")

    print("Starting training")
    global_step = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_lives_loss = 0.0
        running_loss = 0.0
        running_lives_loss = 0.0
        running_count = 0

        print(f"Starting epoch {epoch}...")

        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for observations, actions, lives, lengths in steps_bar:
            observations = observations.to(device)      # (B, seq_len, C, H, W)
            actions = actions.to(device)
            lives = lives.to(device)
            lengths = lengths.to(device)
            
            batch_size, seq_len = observations.shape[:2]

            latents = vae.reparameterize(*vae.encode(observations.view(batch_size*seq_len, *observations.shape[2:])))     # (b*seq_len, z_size)
            latents = latents.view(batch_size, seq_len, -1)
            
            input_latents = latents[:, :-1, :]   # Get all sequences except last for RNN outputs
            output_latents = latents[:, 1:, :]   # Get all sequences except first for ground truth
            actions = actions[:, :-1, :]
            output_lives = lives[:, 1:]

            weights, mus, sigmas, lives_logits, hiddens = model(input_latents, actions)
            latent_loss = rnn_loss(weights, mus, sigmas, hiddens, lengths, output_latents)
            batch_lives_loss = lives_loss(lives_logits, output_lives, lengths)
            loss = latent_loss + batch_lives_loss


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            
            batch_loss = loss.item()
            batch_lives_loss_value = batch_lives_loss.item()
            batch_samples = observations.size(0)
            
            running_loss += batch_loss * batch_samples
            running_lives_loss += batch_lives_loss_value * batch_samples
            running_count += batch_samples
            
            epoch_loss += batch_loss * batch_samples
            epoch_lives_loss += batch_lives_loss_value * batch_samples

            # steps_bar.write(f"Finished training step {global_step}")
            if global_step % log_every == 0:
                avg_loss = running_loss / running_count
                avg_lives_loss = running_lives_loss / running_count

                # Subsampled evaluation for step-level checkpointing
                test_metrics = evaluate(model, vae, test_dl, device, max_batches=num_eval_batches)

                log_metrics({
                    "train/loss": avg_loss,
                    "train/lives_loss": avg_lives_loss,
                    "test/loss": test_metrics["loss"],
                    "test/lives_loss": test_metrics["lives_loss"],
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)

                log_predictions(
                    test_metrics["sample_originals"],
                    test_metrics["sample_predictions"],
                    test_metrics["sample_ground_truths"],
                    test_metrics["sample_actions"],
                    step=global_step,
                )

                # Step-wise checkpointing
                save_checkpoint(
                    model, optimizer,
                    save_dir=step_ckpt_dir,
                    filename=f"vae_step_{global_step:06d}.pt",
                    metadata={"global_step": global_step, "epoch": epoch,
                              "train_loss": avg_loss, "test_loss": test_metrics["loss"],
                              "train_lives_loss": avg_lives_loss, "test_lives_loss": test_metrics["lives_loss"]},
                )

                steps_bar.write(f"Step {global_step} (epoch {epoch}) — train: {avg_loss:.4f}  "
                                f"train_lives: {avg_lives_loss:.4f}  test: {test_metrics['loss']:.4f}  "
                                f"test_lives: {test_metrics['lives_loss']:.4f}")

                running_loss = 0.0
                running_lives_loss = 0.0
                running_count = 0

                model.train()

        # Epoch level checkpointing and evaluation (evaluation over the whole evaluation set)
        n_train = len(train_dl.dataset)
        avg_epoch_loss = epoch_loss / n_train
        avg_epoch_lives_loss = epoch_lives_loss / n_train

        print(f"\nEpoch {epoch} training done (step {global_step}) — "
              f"train_loss: {avg_epoch_loss:.4f}  train_lives_loss: {avg_epoch_lives_loss:.4f}")

        print("Running full evaluation on test set...")
        test_metrics = evaluate(model, vae, test_dl, device)

        log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_lives_loss": avg_epoch_lives_loss,
            "test/epoch_loss": test_metrics["loss"],
            "test/epoch_lives_loss": test_metrics["lives_loss"],
            "epoch": epoch,
            "global_step": global_step,
        }, step=global_step)

        log_predictions(
            test_metrics["sample_originals"],
            test_metrics["sample_predictions"],
            test_metrics["sample_ground_truths"],
            test_metrics["sample_actions"],
            step=global_step,
        )

        save_checkpoint(
            model, optimizer,
            save_dir=epoch_ckpt_dir,
            filename=f"rnn_epoch_{epoch:03d}.pt",
            metadata={"epoch": epoch, "global_step": global_step,
                      "train_loss": avg_epoch_loss, "test_loss": test_metrics["loss"],
                      "train_lives_loss": avg_epoch_lives_loss,
                      "test_lives_loss": test_metrics["lives_loss"]},
        )

        print(f"Epoch {epoch} complete — test_loss: {test_metrics['loss']:.4f}  "
              f"test_lives_loss: {test_metrics['lives_loss']:.4f}")

        model.train()


@torch.no_grad()
def evaluate(model, vae, test_dl, device, max_batches=None):
    """Evaluation function. `max_batches` specifies the number of batches for evaluation, for when I need step-level checkpointing"""
    model.eval()

    total_loss = 0.0
    total_lives_loss = 0.0
    n_samples = 0
    samples = None

    total_batches = min(max_batches, len(test_dl)) if max_batches is not None and max_batches >= 0 else len(test_dl)
    eval_bar = tqdm(test_dl, desc="Evaluating", total=total_batches)

    for observations, actions, lives, lengths in eval_bar:
        observations = observations.to(device)
        actions = actions.to(device)
        lives = lives.to(device)
        lengths = lengths.to(device)

        batch_size, seq_len = observations.shape[:2]
        flat_observations = observations.view(
            batch_size * seq_len, *observations.shape[2:]
        )
        latents = vae.reparameterize(*vae.encode(flat_observations))
        latents = latents.view(batch_size, seq_len, -1)

        input_latents = latents[:, :-1, :]
        output_latents = latents[:, 1:, :]
        input_actions = actions[:, :-1, :]
        output_lives = lives[:, 1:]

        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            weights, mus, sigmas, lives_logits, hiddens = model(input_latents, input_actions)
            latent_loss = rnn_loss(weights, mus, sigmas, hiddens, lengths, output_latents)
            batch_lives_loss = lives_loss(lives_logits, output_lives, lengths)
            loss = latent_loss + batch_lives_loss

        batch_loss = loss.item()
        batch_lives_loss_value = batch_lives_loss.item()
        batch_samples = observations.size(0)
        total_loss += batch_loss * batch_samples
        total_lives_loss += batch_lives_loss_value * batch_samples
        n_samples += batch_samples

        if samples is None:
            samples = build_prediction_samples(vae, observations, actions, weights, mus, lengths)

        average_loss = total_loss / n_samples
        average_lives_loss = total_lives_loss / n_samples
        eval_bar.set_postfix(loss=f"{average_loss:.4f}", lives_loss=f"{average_lives_loss:.4f}")

        if max_batches and eval_bar.n >= max_batches:
            break

    eval_bar.close()

    average_loss = total_loss / n_samples
    average_lives_loss = total_lives_loss / n_samples
    print(f"Evaluated on {n_samples} samples — loss: {average_loss:.4f}  "
          f"lives_loss: {average_lives_loss:.4f}")

    return {
        "loss": average_loss,
        "lives_loss": average_lives_loss,
        "sample_originals": samples["org"],
        "sample_predictions": samples["pred"],
        "sample_ground_truths": samples["gt"],
        "sample_actions": samples["act"],
    }


if __name__ == "__main__":
    DATA_DIR = "./data/"
    vae_weights_path = "./checkpoints/vae/epochs/vae_epoch_025.pt"
    latents_dim = 128
    run_name = input("Enter run name (Optional): ").strip() or None
    
    train(DATA_DIR, vae_weights_path, run_name=run_name, latents_dim=latents_dim, epochs=5, batch_size=2, log_every=100, num_eval_batches=50)
