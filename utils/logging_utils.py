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


def save_checkpoint(
    model,
    optimizer,
    save_dir: str | Path,
    filename: str,
    metadata: dict | None = None,
    max_keep: int = 5,
):
    print(f"Saving a model checkpoint at {Path(save_dir) / filename} with the metadata {metadata}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if metadata:
        checkpoint.update(metadata)
    
    path = save_dir / filename
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

    # Rotate: keep only the most recent max_keep checkpoints
    existing = sorted(save_dir.glob("*.pt"))
    for old in existing[:-max_keep]:
        old.unlink()
    
    return path


def load_checkpoint(path: str | Path, model, optimizer=None):
    checkpoint = torch.load(path, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Return metadata (everything except model/optimizer state)
    return {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict")}
