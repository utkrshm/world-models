# Notes

## Playing with the environment

- Environment is continuous, Box environment with default shape (210, 160, 3).
- `videos` directory contains sample episodes that we recorded.
- Average reward over a 5-episode rollout for a random agent is ~200
- Best performance on Ms. Pacman (by an agent, as descripted in the OpenAI Gym leaderboard) ever is around 5500 in a 100-episode interval, so let's say that the aim is ~3000 if we can achieve it.
- Authors have used VizDoom and CarRacing, and these environments in the beginning are a lot more stochastic than MsPacman, in the sense that MsPacman's reset state is deterministic, it doesn't change. The same can't be said for the other two environments.
- As it turns out, for MsPacman, the action space is discrete, unlike the box action space for VizDoom or CarRacing. So, a (9,) vector has to be output of the controller.
- **NoOp Problem:** On every reset, there's a certain number of frames until which the agent doesn't move despite being repeatedly given control signals. This might dilute my training set, so I need to come up with a way to remove those initial frames.

## Design thoughts
- I wanna penalize the agent a little bit for losing a life in the environment. Seems a bit harsh, i get it, but then it's better to do that, because then the agent doesn't get reset back to the center, losing the current progress in a particular path. Either I can do it by adding a reward term, or if we follow the approach from the paper and create a virtual wrapper around the mdn-rnn model, then we can do what they did and make the M model predict a discrete "lives" count as well... this way atleast the information will still be encoded into the model.
- **Data Storage:** Not keeping it too complicated, one `npz` file and a single `json` for the same episode, with the same name.
  - Changed this later to multiple different `npy` files for the same episode, storing different aspects of the episodes (observations, actions, rewards, lives) because I wanted efficient retrieval during data loading because the data loading on each batch was taking wayyyy too much time (got a 7x speedup in training time because of this tiny change).


## Areas to further explore / Future reading after this paper
- *Replay Comes Of Age (Foster, 2017)*
- Bayesian Neural Networks
[x] Mixture Density Models (covered a little for this implementation as well, but still, it's a fascinating concept)
- Gaussian Proceses to learn system dynamics
- PILCO


## Issue with the VAE not learning dynamics properly
- M module doesn't seem to be training properly, it starts at 90, then quickly drops to 80 and stays constant... Seems like it's not learning anything atp
- Upon further inspection of the VAE reconstruction logs, the VAE is not learning anything at all... There's absolutely no learning in the VAE. All the reconstructions look exactly the same. This is further verified by probing the latents of the VAE while training, and the value of the KL divergence becoming infinitesimal. The logging on the VAE is horrible. Instead of using a alternate-image approach, using a subplot will be better, as the headings will then be bigger atleast.
  - Even after changing the spatial dimensions to include a 4x4 feature map and correcting the imbalance of weight count between the encoder and the decoder, the character sprites (PacMan, ghosts and the power pellet) don't render at all. It's all just a plain environment, without Pacman, the ghosts or the power pellet. The KL divergence starts going down to almost 0 every time. **The main observation is that almost all of the contribution of the loss is from the reconstruction error, and not the KL divergence**.
    - Maybe trying out normalizing the reconstruction loss might help...

- Didn't try the normalizing reconstruction loss thing, coz that would've been useless... Did try KL annealing with difff weights, removing the AutoScaler (coz I thought that might be giving issues), changing the spatial configuration of the architecture (tried 4x4 feature maps and 8x8 feature maps from the default 2x2 in the original paper) and fixed my mistake of the horrible weight imbalance between the decoder and encoder (in old code, decoder had 81% of the weights of the architecture lol)...
  
- Currently trying to train an AutoEncoder (without reparameterization) on simple reconstruction loss (mse only) to see if the architecture can even learn to reconstruct the sprites / HUD properly.
  - Overfitting when perfectly, over time (200 epochs) it managed to perfectly replicate the given sample
  - Normal training went the same as training the VAE; it reconstructed the maze properly, but only the maze. Information about the sprites was entirely missing...
- Tried to see if encoder-to-latent compression was the issue, so I ran normal training over different models with different latent sizes, that also seems to have run into the same problem even with latents as huge as 1024 dims... So that means that the architecture is just incapable of being able to learn the finer details in the maze... But at the same time if that were true, then it shouldn't have been able to mug up the single frame... Maybe the problem is not enough training time? 
  - After re-running normal training on the AE (`train_ae.py` script) for 10 epochs (10x more than the other tests), the results are the same lol. So doesn't seem like the training size is the issue. The behaviour evident in the first epoch almost flatlines in the later epochs...
- Tried retraining the VAE for a longer duration (10 epochs), hoping that it might solve the issue, coz the AE was performing better at reconstruction because of a longer training run... But then the KLD becomes horrible (hits a false bottom around 0.3 around step 700, but then climbs back up to 2 and flatlines around 2-ish even after 10 epochs). For sure training time isn't the issue, coz even tho the recon loss goes down, the KLD doesn't
  - Even despite all odds, tried training the M module on this, but then, as expected, it wasn't that useful... Loss started at 180 in the first 100 steps, stayed at 180 around the 10th epoch as well (I was really the Grokking effect would occcur here, but then I guess not, also doesn't help that training is soo slow on my 4GB VRAM laptop lol)

**Conclusion:** I think I need to somehow include in the reconstruction loss the reconstruction of the sprites. Need to read up on how to do that.