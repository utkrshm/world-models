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
- **Data Storage:** Not keeping it too complicated, one `npy` file and a single `json` for the same episode, with the same name. 

## Issues
- M module doesn't seem to be training properly, it starts at 90, then quickly drops to 80 and stays constant... Seems like it's not learning anything atp
- Upon further inspection of the VAE reconstruction logs, the VAE is not learning anything at all... There's absolutely no learning in the VAE. All the reconstructions look exactly the same. This is further verified by probing the latents of the VAE while training, and the value of the KL divergence becoming infinitesimal. The logging on the VAE is horrible. Instead of using a alternate-image approach, using a subplot will be better, as the headings will then be bigger atleast.
  - Even after changing the spatial dimensions to include a 4x4 feature map and correcting the imbalance of weight count between the encoder and the decoder, the character sprites (PacMan, ghosts and the power pellet) don't render at all. It's all just a plain environment, without Pacman, the ghosts or the power pellet. The KL divergence starts going down to almost 0 every time. **The main observation is that almost all of the contribution of the loss is from the reconstruction error, and not the KL divergence**.
    - Maybe trying out normalizing the reconstruction loss might help


## Areas to further explore / Future reading after this paper
- *Replay Comes Of Age (Foster, 2017)*
- Bayesian Neural Networks
[x] Mixture Density Models (covered a little for this implementation as well, but still, it's a fascinating concept)
- Gaussian Proceses to learn system dynamics
- PILCO