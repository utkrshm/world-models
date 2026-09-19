import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from models.autoenc import AutoEncoder
from utils.data_utils import VAEDataset
from utils.logging_utils import init_wandb, log_metrics


LATENT_SIZES = (128, 256, 512, 1024)
WANDB_PROJECT_NAME = "world-models-autoenc-latent-size"


def get_loaders(data_dir, batch_size=16, num_workers=2, shuffle=True, test_pct=0.2):
    ds = VAEDataset(data_dir)

    train_ds, test_ds = random_split(ds, (1-test_pct, test_pct))
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_dl, test_dl


def reconstruction_loss(x_recon, x):
    return F.mse_loss(x_recon, x, reduction="sum")


def build_reconstruction_logs(evaluation_metrics):
    """Build prefixed reconstruction image logs for every latent size."""
    import matplotlib.pyplot as plt

    logs = {}
    for latent_size, metrics in evaluation_metrics.items():
        originals = metrics["sample_originals"]
        reconstructions = metrics["sample_reconstructions"]
        n = min(8, originals.size(0))
        images = []

        for index in range(n):
            figure, axes = plt.subplots(1, 2, figsize=(8, 4))
            original = originals[index].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            reconstruction = reconstructions[index].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

            axes[0].imshow(original)
            axes[0].set_title("Original")
            axes[1].imshow(reconstruction)
            axes[1].set_title(f"Latent size {latent_size}")
            for axis in axes:
                axis.axis("off")

            figure.tight_layout()
            images.append(wandb.Image(figure, caption=f"latent_{latent_size}_reconstruction_{index}"))
            plt.close(figure)

        logs[f"reconstructions/latent_{latent_size}"] = images

    return logs


def evaluate(model, test_dl, device, max_batches=None):
    """Evaluate one model, optionally limiting the number of test batches."""
    model.eval()
    total_recon = 0.0
    n_samples = 0
    sample_originals = None
    sample_reconstructions = None

    n_batches = min(max_batches, len(test_dl)) if max_batches else len(test_dl)
    eval_bar = tqdm(test_dl, desc="Evaluating", total=n_batches)

    with torch.no_grad():
        for batch in eval_bar:
            batch = batch.to(device)
            recon = model(batch)
            recon_loss = reconstruction_loss(recon, batch)

            total_recon += recon_loss.item()
            n_samples += batch.size(0)

            if sample_originals is None:
                sample_originals = batch.detach()
                sample_reconstructions = recon.detach()

            eval_bar.set_postfix(recon_loss=f"{total_recon / n_samples:.4f}")

            if max_batches and eval_bar.n >= max_batches:
                break

    eval_bar.close()
    average_recon_loss = total_recon / n_samples
    return {
        "recon_loss": average_recon_loss,
        "sample_originals": sample_originals,
        "sample_reconstructions": sample_reconstructions,
    }


def evaluate_models(models, test_dl, device, max_batches):
    return {
        latent_size: evaluate(model, test_dl, device, max_batches=max_batches)
        for latent_size, model in models.items()
    }


def train(data_dir, run_name, epochs=1, batch_size=16, num_workers=0, lr=1e-3,
          log_every=100, num_eval_batches=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(epochs=epochs, batch_size=batch_size, lr=lr, device=str(device),
                  log_every=log_every, num_eval_batches=num_eval_batches,
                  latent_sizes=list(LATENT_SIZES))
    init_wandb(config, WANDB_PROJECT_NAME, run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, batch_size=batch_size, num_workers=num_workers)

    print("Loading models to device...")
    models = {latent_size: AutoEncoder(latent_size).to(device) for latent_size in LATENT_SIZES}
    optimizers = {latent_size: optim.Adam(model.parameters(), lr=lr)
                  for latent_size, model in models.items()}

    global_step = 0
    running_losses = {latent_size: 0.0 for latent_size in LATENT_SIZES}
    running_count = 0

    print(f"Starting training for latent sizes: {LATENT_SIZES}")
    for epoch in range(1, epochs + 1):
        for model in models.values():
            model.train()

        epoch_losses = {latent_size: 0.0 for latent_size in LATENT_SIZES}
        print(f"Starting epoch {epoch}...")

        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for batch in steps_bar:
            batch = batch.to(device)
            global_step += 1

            for latent_size in LATENT_SIZES:
                model = models[latent_size]
                recon = model(batch)
                loss = reconstruction_loss(recon, batch)

                optimizers[latent_size].zero_grad()
                loss.backward()
                optimizers[latent_size].step()

                running_losses[latent_size] += loss.item()
                epoch_losses[latent_size] += loss.item()

            running_count += batch.size(0)

            if global_step % log_every == 0:
                evaluation_metrics = evaluate_models(
                    models,
                    test_dl,
                    device,
                    max_batches=num_eval_batches,
                )
                metrics = {
                    "epoch": epoch,
                    "global_step": global_step,
                }
                for latent_size in LATENT_SIZES:
                    prefix = f"latent_{latent_size}"
                    metrics[f"train/{prefix}/recon_loss"] = (
                        running_losses[latent_size] / running_count
                    )
                    metrics[f"test/{prefix}/recon_loss"] = evaluation_metrics[
                        latent_size
                    ]["recon_loss"]

                log_metrics({**metrics, **build_reconstruction_logs(evaluation_metrics)}, step=global_step)

                summary = "  ".join(f"{latent_size}: {evaluation_metrics[latent_size]['recon_loss']:.4f}" for latent_size in LATENT_SIZES)
                steps_bar.write(f"Step {global_step} (epoch {epoch}) - test: {summary}")

                running_losses = {latent_size: 0.0 for latent_size in LATENT_SIZES}
                running_count = 0
                for model in models.values():
                    model.train()

        n_train = len(train_dl.dataset)
        average_epoch_losses = {latent_size: epoch_losses[latent_size] / n_train for latent_size in LATENT_SIZES}
        evaluation_metrics = evaluate_models(
            models,
            test_dl,
            device,
            max_batches=num_eval_batches,
        )

        metrics = {
            "epoch": epoch,
            "global_step": global_step,
        }
        for latent_size in LATENT_SIZES:
            prefix = f"latent_{latent_size}"
            metrics[f"train/epoch/{prefix}/recon_loss"] = average_epoch_losses[latent_size]
            metrics[f"test/epoch/{prefix}/recon_loss"] = evaluation_metrics[latent_size]["recon_loss"]

        log_metrics({**metrics, **build_reconstruction_logs(evaluation_metrics)}, step=global_step)

        summary = "  ".join(f"{latent_size}: {evaluation_metrics[latent_size]['recon_loss']:.4f}" for latent_size in LATENT_SIZES)
        print(f"Epoch {epoch} complete - test reconstruction loss: {summary}\n")


if __name__ == "__main__":
    DATA_DIR = "./data/"
    
    run_name = input("Enter run name (Optional): ")
    
    train(DATA_DIR, run_name=run_name, batch_size=128, log_every=100, num_eval_batches=50)
