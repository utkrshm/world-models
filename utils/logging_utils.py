from pathlib import Path

import torch
import wandb

def init_wandb(config: dict, project: str, run_name: str | None = None):
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def log_metrics(metrics: dict, step: int):
    wandb.log(metrics, step=step)


def log_reconstructions(originals, reconstructions, step: int, num_images: int = 8):
    import matplotlib.pyplot as plt
    
    n = min(num_images, originals.size(0))
    
    images = []
    for i in range(n):
        original = originals[i].clamp(0, 1).permute(1, 2, 0).detach().float().cpu().numpy()
        reconstruction = reconstructions[i].clamp(0, 1).permute(1, 2, 0).detach().float().cpu().numpy()

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].imshow(original)
        axes[0].set_title("Original")
        
        axes[1].imshow(reconstruction)
        axes[1].set_title("Reconstruction")

        for ax in axes: ax.axis("off")
        figure.tight_layout()

        images.append(wandb.Image(figure, caption=f"reconstruction_{i}"))
        
        plt.close(figure)

    wandb.log({"reconstructions": images}, step=step)
    plt.close("all")


def log_predictions(orig, pred, gt, act, step: int, num_images: int = 8):
    import matplotlib.pyplot as plt

    n = min(num_images, orig.size(0), pred.size(0), gt.size(0))
    images = []

    for i in range(n):
        action = act[i]
        if torch.is_tensor(action):
            action = action.argmax().item() if action.ndim else action.item()

        figure, axes = plt.subplots(1, 3, figsize=(12, 4))
        observations = (
            orig[i],
            pred[i],
            gt[i],
        )
        titles = (
            "Timestamp t",
            f"Prediction t+1 (action {action})",
            f"Ground truth t+1 (action {action})",
        )

        for axis, observation, title in zip(axes, observations, titles):
            image = observation.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()
            axis.imshow(image)
            axis.set_title(title)
            axis.axis("off")

        figure.tight_layout()
        images.append(wandb.Image(figure, caption=f"prediction_{i}"))
        plt.close(figure)

    wandb.log({"predictions": images}, step=step)
    plt.close("all")


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
