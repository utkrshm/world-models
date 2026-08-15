# Notes

## Playing with the environment

- Environment is continuous, Box environment with default shape (210, 160, 3).
- `videos` directory contains sample episodes that we recorded.
- Average reward over a 5-episode rollout for a random agent is ~200
- Best performance on Ms. Pacman (by an agent, as descripted in the OpenAI Gym leaderboard) ever is around 5500 in a 100-episode interval, so let's say that the aim is ~3000 if we can achieve it.
- Authors have used VizDoom and CarRacing, and these environments in the beginning are a lot more stochastic than MsPacman, in the sense that MsPacman's reset state is deterministic, it doesn't change. The same can't be said for the other two environments.

## Design thoughts
- I wanna penalize the agent a little bit for losing a life in the environment. Seems a bit harsh, i get it, but then it's better to do that, because then the agent doesn't get reset back to the center, losing the current progress in a particular path. Either I can do it by adding a reward term, or if we follow the approach from the paper and create a virtual wrapper around the mdn-rnn model, then we can do what they did and make the M model predict a discrete "lives" count as well... this way atleast the information will still be encoded into the model.