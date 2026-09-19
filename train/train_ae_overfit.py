import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.autoenc import AutoEncoder
from utils.data_utils import VAEDataset
from utils.logging_utils import (
    init_wandb,
    log_metrics,
    log_reconstructions,
)

WANDB_PROJECT_NAME = "world-models-autoenc"

def get_loaders(data_dir, num_workers=0):
    """Return identical train/test loaders containing the first sample only."""
    dataset = VAEDataset(data_dir)
    if len(dataset) == 0:
        raise ValueError(f"No observations found in {data_dir}")

    sample = dataset[0].unsqueeze(0)
    loader = DataLoader(
        sample,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    return loader, loader


def reconstruction_loss(x_recon, x):
    return F.mse_loss(x_recon, x, reduction="sum")


def train(data_dir, run_name, epochs=100, num_workers=0, lr=1e-3,
          log_every=1, num_eval_batches=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = dict(epochs=epochs, batch_size=1, lr=lr, device=str(device), log_every=log_every)
    
    init_wandb(config, WANDB_PROJECT_NAME, run_name)

    print("Getting data loaders...")
    train_dl, test_dl = get_loaders(data_dir, num_workers=num_workers)

    print("Loading the model to device...")
    model = AutoEncoder(128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # use_amp = device.type == "cuda"
    # scaler = torch.GradScaler("cuda", enabled=use_amp)

    global_step = 0
    running_loss = 0.0
    running_count = 0
    
    print("Starting training")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        print(f"Starting epoch {epoch}...")

        steps_bar = tqdm(train_dl, desc=f"Epoch {epoch}/{epochs}")
        for batch in steps_bar:
            batch = batch.to(device)
            global_step += 1

            recon = model(batch)
            loss = reconstruction_loss(recon, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_count += batch.size(0)
            epoch_loss += loss.item()

            # steps_bar.write(f"Finished training step {global_step}")
            if global_step % log_every == 0:
                avg_loss = running_loss / running_count

                # Evaluate a small validation subset for step-wise logging.
                test_metrics = evaluate(
                    model,
                    test_dl,
                    device,
                    max_batches=num_eval_batches,
                )

                log_metrics({
                    "train/loss": avg_loss,
                    "train/recon_loss": avg_loss,
                    "test/loss": test_metrics["recon_loss"],
                    "test/recon_loss": test_metrics["recon_loss"],
                    "epoch": epoch,
                    "global_step": global_step,
                }, step=global_step)

                log_reconstructions(
                    test_metrics["sample_originals"], 
                    test_metrics["sample_reconstructions"], 
                    step=global_step,
                )

                steps_bar.write(f"Step {global_step} (epoch {epoch}) - train: {avg_loss:.4f}  test: {test_metrics['recon_loss']:.4f}")

                running_loss = 0.0
                running_count = 0

                model.train()

        # Evaluate over the full validation set at the end of each epoch.
        n_train = len(train_dl.dataset)
        avg_epoch_loss = epoch_loss / n_train

        print(f"\nEpoch {epoch} training done (step {global_step}) - "
              f"recon_loss: {avg_epoch_loss:.4f}")

        print("Running full evaluation on test set...")
        test_metrics = evaluate(model, test_dl, device)

        log_metrics({
            "train/epoch_loss": avg_epoch_loss,
            "train/epoch_recon_loss": avg_epoch_loss,
            "test/epoch_loss": test_metrics["recon_loss"],
            "test/epoch_recon_loss": test_metrics["recon_loss"],
            "epoch": epoch,
            "global_step": global_step,
        }, step=global_step)

        log_reconstructions(
            test_metrics["sample_originals"],
            test_metrics["sample_reconstructions"],
            step=global_step,
        )

        print(f"Epoch {epoch} complete - test_recon: {test_metrics['recon_loss']:.4f}\n")

        model.train()


@torch.no_grad()
def evaluate(model, test_dl, device, max_batches=None):
    """Evaluate reconstruction loss, optionally over only the first batches."""
    model.eval()

    total_recon = 0.0
    n_samples = 0
    sample_originals = None
    sample_reconstructions = None

    n_batches = min(max_batches, len(test_dl)) if max_batches else len(test_dl)
    eval_bar = tqdm(test_dl, desc="Evaluating", total=n_batches)

    for batch in eval_bar:
        batch = batch.to(device)

        recon = model(batch)
        recon_loss = reconstruction_loss(recon, batch)

        total_recon += recon_loss.item()
        n_samples += batch.size(0)

        # Grab the first batch we see for visualization
        if sample_originals is None:
            sample_originals = batch.detach()
            sample_reconstructions = recon.detach()

        eval_bar.set_postfix(recon_loss=f"{total_recon / n_samples:.4f}")

        if max_batches and eval_bar.n >= max_batches:
            break

    eval_bar.close()

    average_recon_loss = total_recon / n_samples
    print(f"Evaluated on {n_samples} samples - recon_loss: {average_recon_loss:.4f}")

    return {
        "recon_loss": average_recon_loss,
        "sample_originals": sample_originals,
        "sample_reconstructions": sample_reconstructions,
    }


if __name__ == "__main__":
    DATA_DIR = "./data/"
    
    run_name = input("Enter run name (Optional): ")
    
    train(DATA_DIR, run_name=run_name, epochs=200, log_every=1, num_eval_batches=1)
