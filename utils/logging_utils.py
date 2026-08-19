from pathlib import Path

import torch
import wandb

def init_wandb(config: dict, project: str, run_name: str | None = None):
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def log_metrics(metrics: dict, step: int):
    wandb.log(metrics, step=step)


def log_reconstructions(originals, reconstructions, step: int, num_images: int = 8):
    n = min(num_images, originals.size(0))
    
    images = []
    for i in range(n):
        # Tensors are (C, H, W), clamp to valid range before logging
        orig = originals[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        recon = reconstructions[i].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        
        images.append(wandb.Image(orig, caption=f"orig_{i}"))
        images.append(wandb.Image(recon, caption=f"recon_{i}"))
    
    wandb.log({"reconstructions": images}, step=step)


def save_checkpoint(model, optimizer, epoch: int, train_loss: float, test_loss: float, save_dir: str | Path):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "test_loss": test_loss,
    }
    
    path = save_dir / f"vae_epoch_{epoch:03d}.pt"
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")
    
    return path


def load_checkpoint(path: str | Path, model, optimizer=None):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return checkpoint["epoch"]
