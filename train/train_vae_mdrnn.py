"""Joint training of the VAE and Memory modules."""
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.vae import VAE
from models.mdrnn import Memory
from utils.data_utils import RNNDataset
from utils.logging_utils import (
    init_wandb,
    load_checkpoint,
    log_metrics,
    log_reconstructions,
    log_predictions,
    save_checkpoint,
)

import matplotlib
matplotlib.use("Agg")

CHECKPOINT_DIR = "./checkpoints/joint-training/"
WANDB_PROJECT_NAME = "world-models-joint-train"


def get_loaders(data_dir, batch_size=16, num_workers=0, shuffle=True, test_pct=0.2):
    
    def collate_fn(batch):
        # batch = [(obs[0], acts[0]), ..., (obs[i], acts[i])]
        observations = [ep[0] for ep in batch]
        actions = [ep[1] for ep in batch]
        
        lengths = torch.tensor([len(obs) for obs in observations])
        
        observations = pad_sequence(observations, batch_first=True)
        actions = pad_sequence(actions, batch_first=True)
        
        # Return padded sequence, along with original sequnence length
        return observations, actions, lengths
        
    ds = RNNDataset(data_dir)
    
    train_ds, test_ds = random_split(ds, (1-test_pct, test_pct))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn)
    
    return train_dl, test_dl


def build_prediction_samples(vae, observations, actions, weights, mus, lengths, reconstructions, num_seqs=8, max_frames=16):
    take = min(num_seqs, observations.size(0))
    max_transitions = min(max_frames, weights.size(1))
    frame_shape = observations.shape[2:]

    observations = observations[:take, : max_transitions + 1]
    actions = actions[:take, :max_transitions]
    weights = weights[:take, :max_transitions]
    mus = mus[:take, :max_transitions]
    reconstructions = reconstructions[:take, :max_transitions]

    transition_lengths = (lengths[:take] - 1).clamp(min=0, max=max_transitions)
    valid = torch.arange(max_transitions, device=observations.device).unsqueeze(0)
    valid = valid < transition_lengths.unsqueeze(1)

    pred_latents = (weights.float().unsqueeze(3) * mus.float()).sum(dim=2)
    pred = vae.decode(pred_latents.flatten(0, 1))
    pred = pred.view(take, max_transitions, *frame_shape)

    return {
        "org": observations[:, :-1][valid],
        "recon": reconstructions[valid],
        "pred": pred[valid],
        "gt": observations[:, 1:][valid],
        "act": actions[valid],
    }


def save_joint_checkpoints(vae, memory, optimizer, save_dir, filename, metadata):
    save_checkpoint(vae, optimizer, os.path.join(save_dir, "vae"), filename, metadata)
    save_checkpoint(memory, optimizer, os.path.join(save_dir, "memory"), filename, metadata)


def load_run(checkpoint_dir, filename, latents_dim, hiddens_dim, actions_dim,
             n_mixtures, lr=1e-3, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the models to device...")
    vae = VAE(latents_dim).to(device)
    memory = Memory(latents_dim, hiddens_dim, actions_dim, n_mixtures).to(device)
    optimizer = optim.Adam(
        list(vae.parameters()) + list(memory.parameters()),
        lr=lr,
    )

    vae_checkpoint = os.path.join(checkpoint_dir, "vae", filename)
    memory_checkpoint = os.path.join(checkpoint_dir, "memory", filename)

    vae_metadata = load_checkpoint(vae_checkpoint, vae, optimizer)
    memory_metadata = load_checkpoint(memory_checkpoint, memory, optimizer)

    metadata = {**vae_metadata, **memory_metadata}
    return vae, memory, optimizer, metadata


# ===== LOSS FUNCTIONS =====
def vae_loss(x_recon, x, mu, logvar):
    # Changing loss scaling to be mean reduction per-frame
    
    # Per-frame reconstruction loss
    recon_loss = F.mse_loss(x_recon, x, reduction="none")
    recon_loss = recon_loss.flatten(1).sum(dim=1).mean()

    # Per-frame KL
    kld = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld.sum(dim=1).mean()

    return recon_loss + kld, recon_loss, kld

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


# ===== TRAIN FUNCTION ====
def train(data_dir, run_name, latents_dim, hiddens_dim, actions_dim, n_mixtures,
          epochs=1, batch_size=16, num_workers=0, lr=1e-3, log_every=1000,
          num_eval_batches=10, lambda_recon=1.0, beta_kl=1.0, lambda_rnn=0.6,
          checkpoint_path=None, checkpoint_filename=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = {
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": str(device),
        "log_every": log_every,
        "latents_dim": latents_dim,
        "hiddens_dim": hiddens_dim,
        "actions_dim": actions_dim,
        "n_mixtures": n_mixtures,
        "lambda_recon": lambda_recon,
        "beta_kl": beta_kl,
        "lambda_rnn": lambda_rnn,
        "checkpoint_path": checkpoint_path,
        "checkpoint_filename": checkpoint_filename,
    }
    init_wandb(config, WANDB_PROJECT_NAME, run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    print("Loading the models to device...")
    if (checkpoint_path is None) != (checkpoint_filename is None):
        raise ValueError("checkpoint_path and checkpoint_filename must be provided together")

    if checkpoint_path is not None:
        vae, memory, optimizer, checkpoint_metadata = load_run(
            checkpoint_path,
            checkpoint_filename,
            latents_dim,
            hiddens_dim,
            actions_dim,
            n_mixtures,
            lr=lr,
            device=device,
        )
        global_step = checkpoint_metadata.get("global_step", 0)
        start_epoch = checkpoint_metadata.get("epoch", 0) + 1
        print(f"Continuing from epoch {start_epoch} at step {global_step}...")
    else:
        vae = VAE(latents_dim).to(device)
        memory = Memory(latents_dim, hiddens_dim, actions_dim, n_mixtures).to(device)
        optimizer = optim.Adam(
            list(vae.parameters()) + list(memory.parameters()),
            lr=lr,
        )
        global_step = 0
        start_epoch = 1

    step_ckpt_dir = os.path.join(CHECKPOINT_DIR, "steps")
    epoch_ckpt_dir = os.path.join(CHECKPOINT_DIR, "epochs")
    running = {"loss": 0.0, "recon_loss": 0.0, "kld": 0.0, "rnn_loss": 0.0, "count": 0}

    print("Starting training")
    for epoch in range(start_epoch, epochs + 1):
        vae.train()
        memory.train()
        epoch_totals = {"loss": 0.0, "recon_loss": 0.0, "kld": 0.0, "rnn_loss": 0.0}

        print(f"Starting epoch {epoch}...")
        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for observations, actions, lengths in steps_bar:
            observations = observations.to(device)
            actions = actions.to(device)
            lengths = lengths.to(device)
            
            batch_size_actual, seq_len = observations.shape[:2]
            flat_observations = observations.view(batch_size_actual * seq_len, *observations.shape[2:])
            
            # Get the latents for the entire episode
            mu, logvar = vae.encode(flat_observations)
            latents = vae.reparameterize(mu, logvar).view(batch_size_actual, seq_len, -1)
            
            mu = mu.view(batch_size_actual, seq_len, -1)
            logvar = logvar.view(batch_size_actual, seq_len, -1)

            # Define inputs and outputs
            input_latents = latents[:, :-1]
            target_latents = latents[:, 1:].detach()
            input_actions = actions[:, :-1]

            # Get loss components from the VAE
            x_t = observations[:, :-1].reshape(-1, *observations.shape[2:])
            x_recon = vae.decode(input_latents.reshape(-1, input_latents.size(-1)))
            _, recon_loss, kld = vae_loss(
                x_recon,
                x_t,
                mu=mu[:, :-1].reshape(-1, mu.size(-1)),
                logvar=logvar[:, :-1].reshape(-1, logvar.size(-1)),
            )

            # Get MDN-RNN loss
            weights, mus, sigmas, hiddens = memory(input_latents, input_actions)
            dynamics_loss = rnn_loss(
                weights, mus, sigmas, hiddens, lengths, target_latents
            )
            
            # Combine loss
            loss = (
                lambda_recon * recon_loss
                + beta_kl * kld
                + lambda_rnn * dynamics_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            batch_count = observations.size(0)
            
            values = {
                "loss": loss.item(),
                "recon_loss": recon_loss.item(),
                "kld": kld.item(),
                "rnn_loss": dynamics_loss.item(),
            }
            
            for name, value in values.items():
                running[name] += value * batch_count
                epoch_totals[name] += value * batch_count
            running["count"] += batch_count

            if global_step % log_every == 0:
                averages = {name: running[name] / running["count"] for name in values}
                # Subsampled evaluation for step-level checkpointing
                test_metrics = evaluate(
                    vae, memory, test_dl, device, beta_kl,
                    lambda_recon, lambda_rnn, max_batches=num_eval_batches,
                )
                
                log_metrics(
                    {**{f"train/{name}": value for name, value in averages.items()},
                     **{f"test/{name}": test_metrics[name] for name in values},
                     "epoch": epoch, "global_step": global_step},
                    step=global_step,
                )
                
                log_reconstructions(
                    test_metrics["sample_originals"],
                    test_metrics["sample_reconstructions"],
                    step=global_step,
                )
                
                log_predictions(
                    test_metrics["sample_originals"],
                    test_metrics["sample_predictions"],
                    test_metrics["sample_ground_truths"],
                    test_metrics["sample_actions"],
                    step=global_step,
                )
                
                # Step-wise checkpointing
                save_joint_checkpoints(
                    vae, memory, optimizer, step_ckpt_dir,
                    f"joint_step_{global_step:06d}.pt",
                    {"global_step": global_step, "epoch": epoch,
                     "train_loss": averages["loss"], "test_loss": test_metrics["loss"]},
                )
                
                steps_bar.write(
                    f"Step {global_step} (epoch {epoch}) — "
                    f"train: {averages['loss']:.4f}  test: {test_metrics['loss']:.4f}"
                )
                
                running = {"loss": 0.0, "recon_loss": 0.0, "kld": 0.0, "rnn_loss": 0.0, "count": 0}
                
                vae.train()
                memory.train()

        # Epoch level checkpointing and evaluation (evaluation over the whole evaluation set)
        n_train = len(train_dl.dataset)
        epoch_averages = {name: epoch_totals[name] / n_train for name in epoch_totals}
        
        print(f"\nEpoch {epoch} training done (step {global_step}) — "
              f"train_loss: {epoch_averages['loss']:.4f}  "
              f"recon: {epoch_averages['recon_loss']:.4f}  "
              f"kld: {epoch_averages['kld']:.4f}  "
              f"rnn: {epoch_averages['rnn_loss']:.4f}")

        print("Running full evaluation on test set...")
        test_metrics = evaluate(
            vae, memory, test_dl, device, beta_kl,
            lambda_recon, lambda_rnn,
        )
        
        log_metrics(
            {**{f"train/epoch_{name}": value for name, value in epoch_averages.items()},
             **{f"test/epoch_{name}": test_metrics[name] for name in epoch_averages},
             "epoch": epoch, "global_step": global_step},
            step=global_step,
        )
        
        log_reconstructions(test_metrics["sample_originals"], test_metrics["sample_reconstructions"], global_step)
        
        log_predictions(
            test_metrics["sample_originals"], test_metrics["sample_predictions"],
            test_metrics["sample_ground_truths"], test_metrics["sample_actions"], global_step,
        )
        
        save_joint_checkpoints(
            vae, memory, optimizer, epoch_ckpt_dir, f"joint_epoch_{epoch:03d}.pt",
            {"epoch": epoch, "global_step": global_step,
             "train_loss": epoch_averages["loss"], "test_loss": test_metrics["loss"]},
        )
        
        print(f"Epoch {epoch} complete — test_loss: {test_metrics['loss']:.4f}  "
              f"test_recon: {test_metrics['recon_loss']:.4f}  "
              f"test_kld: {test_metrics['kld']:.4f}  "
              f"test_rnn: {test_metrics['rnn_loss']:.4f}\n")


@torch.no_grad()
def evaluate(vae, memory, test_dl, device, beta_kl, lambda_recon, lambda_rnn,
             max_batches=None):
    """Evaluation function. `max_batches` specifies the number of batches for evaluation, for when I need step-level checkpointing"""
    vae.eval()
    memory.eval()
    
    totals = {"loss": 0.0, "recon_loss": 0.0, "kld": 0.0, "rnn_loss": 0.0}
    total_batches = min(max_batches, len(test_dl)) if max_batches else len(test_dl)
    
    n_samples = 0
    samples = None
    
    eval_bar = tqdm(test_dl, desc="Evaluating", total=total_batches)
    for observations, actions, lengths in eval_bar:
        observations = observations.to(device)
        actions = actions.to(device)
        lengths = lengths.to(device)
        
        batch_size_actual, seq_len = observations.shape[:2]
        flat_observations = observations.view(batch_size_actual * seq_len, *observations.shape[2:])

        mu, logvar = vae.encode(flat_observations)
        latents = vae.reparameterize(mu, logvar).view(batch_size_actual, seq_len, -1)
        mu = mu.view(batch_size_actual, seq_len, -1)
        logvar = logvar.view(batch_size_actual, seq_len, -1)
        
        input_latents = latents[:, :-1]
        target_latents = latents[:, 1:]
        input_actions = actions[:, :-1]

        x_t = observations[:, :-1].reshape(-1, *observations.shape[2:])
        x_recon = vae.decode(input_latents.reshape(-1, input_latents.size(-1)))
        _, recon_loss, kld = vae_loss(
            x_recon, x_t,
            mu[:, :-1].reshape(-1, mu.size(-1)),
            logvar[:, :-1].reshape(-1, logvar.size(-1)),
        )
        
        weights, mus, sigmas, hiddens = memory(input_latents, input_actions)
        dynamics_loss = rnn_loss(weights, mus, sigmas, hiddens, lengths, target_latents)
        
        loss = lambda_recon * recon_loss + beta_kl * kld + lambda_rnn * dynamics_loss

        values = {"loss": loss.item(), "recon_loss": recon_loss.item(),
                  "kld": kld.item(), "rnn_loss": dynamics_loss.item()}
        batch_count = observations.size(0)
        for name, value in values.items():
            totals[name] += value * batch_count
        n_samples += batch_count

        if samples is None:
            # Grab the first batch we see for visualization
            prediction_samples = build_prediction_samples(
                vae,
                observations,
                actions,
                weights,
                mus,
                lengths,
                x_recon.view(batch_size_actual, seq_len - 1, *observations.shape[2:]),
            )
            samples = {
                "sample_originals": prediction_samples["org"],
                "sample_reconstructions": prediction_samples["recon"],
                "sample_predictions": prediction_samples["pred"],
                "sample_ground_truths": prediction_samples["gt"],
                "sample_actions": prediction_samples["act"],
            }

        averages = {name: totals[name] / n_samples for name in totals}
        eval_bar.set_postfix(**{name: f"{value:.4f}" for name, value in averages.items()})
        if max_batches and eval_bar.n >= max_batches:
            break

    eval_bar.close()
    averages = {name: totals[name] / n_samples for name in totals}
    print(f"Evaluated on {n_samples} samples — "
          f"loss: {averages['loss']:.4f}  recon: {averages['recon_loss']:.4f}  "
          f"kld: {averages['kld']:.4f}  rnn: {averages['rnn_loss']:.4f}")
    return {**averages, **samples}


if __name__ == "__main__":
    DATA_DIR = "./data/"
    run_name = input("Enter run name (Optional): ").strip() or None
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "steps")
    # checkpoint_filename = "joint_step_002025.pt"

    latents_dim = 128
    hiddens_dim = 512
    actions_dim = 9
    n_mixtures = 5

    lambda_recon = 1.0
    beta_kl = 1.0
    lambda_rnn = 3.0

    train(
        DATA_DIR,
        run_name=run_name,
        latents_dim=latents_dim,
        hiddens_dim=hiddens_dim,
        actions_dim=actions_dim,
        n_mixtures=n_mixtures,
        batch_size=1,
        epochs=10,
        lr=3e-4,
        log_every=200,
        num_eval_batches=25,
        lambda_recon=lambda_recon,
        beta_kl=beta_kl,
        lambda_rnn=lambda_rnn,
        # checkpoint_path=checkpoint_path,
        # checkpoint_filename=checkpoint_filename,
    )
    
