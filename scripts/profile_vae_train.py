from itertools import islice
from pathlib import Path
from time import time

import torch
from torch import optim, profiler
from torch.profiler import ProfilerActivity, profile
from tqdm import tqdm

from models.vae import VAE
from train.train_vae import get_loaders, vae_loss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

start_time = time()
train_dl, test_dl = get_loaders("./data", num_workers=8)
print(f"Time taken to get the loaders: {time()-start_time}")

start_time = time()
model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
print(f"Time taken to initialize model and optimizer: {time()-start_time}")

table_path = Path("./trace_dir/vae_trace002.txt")
trace_path = Path("./trace_dir/vae_trace002.json")
profile_steps = 35

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=profiler.schedule(wait=5, warmup=5, active=5, repeat=2, skip_first=5, skip_first_wait=5),
    on_trace_ready=profiler.tensorboard_trace_handler("./trace_dir")
) as prof:
    for batch in tqdm(islice(train_dl, profile_steps), total=profile_steps):
        batch = batch.to(device)
        
        recon, mu, logvar = model(batch)
        loss, recon_loss, kld = vae_loss(recon, batch, mu, logvar)

        optimizer.zero_grad()
        optimizer.step()
        
        prof.step()
        
    torch.cuda.synchronize()

    # print(f"saving traces... {trace_path}")
    # prof.export_chrome_trace(str(trace_path))
    print(f"saving profiler table ... {table_path}")

    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))