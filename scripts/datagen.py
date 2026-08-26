from pathlib import Path

import ale_py
import cv2
import gymnasium as gym
import numpy as np

NUM_ROLLOUTS = 1000

ENV_NAME = "ALE/MsPacman-v5"
FRAMESKIP_INTERVAL = 4          # Default for the environment is 4 only, kept this for configurations' sake
NOOP_TIME = 65                  # Number of frames for which I've observed that the agent remains static despite being given control signals (actual number is 66, I just want to keep one of these frames for smooth starting state)

SAVE_PATH = Path("data")
SAVE_PATH.mkdir(parents=True, exist_ok=True)


def get_action(env: gym.Env):
    return env.action_space.sample()

def save_data(episode_id: int, obs: list[np.ndarray], acts: list[int], rewards: list[float], lives: list[int]):
    if not obs:
        raise ValueError("Cannot save an empty episode.")

    obs_arr = np.stack(obs, axis=0)
    acts_arr = np.asarray(acts, dtype=np.int64)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    lives_arr = np.asarray(lives, dtype=np.int32)

    np.save(SAVE_PATH / f"{episode_id:05d}_observations.npy", obs_arr)
    np.save(SAVE_PATH / f"{episode_id:05d}_actions.npy", acts_arr)
    np.save(SAVE_PATH / f"{episode_id:05d}_rewards.npy", rewards_arr)
    np.save(SAVE_PATH / f"{episode_id:05d}_lives.npy", lives_arr)

def generate_one_episode(idx, max_steps=1000):
    obs_list = []
    act_list = []
    reward_list = []
    lives_list = []
    
    env = gym.make(
        ENV_NAME, max_steps + NOOP_TIME, 
        frameskip=FRAMESKIP_INTERVAL, 
        render_mode="rgb_array"
    )
    obs, info = env.reset()
    
    # Waste the first few steps, because I don't want training data without any action to corrupt the agent's training
    for _ in range(NOOP_TIME):
        new_obs, reward, terminated, truncated, new_info = env.step(get_action(env))
        obs = new_obs
        info = new_info

        if terminated or truncated:
            return
        
    for _ in range(max_steps):
        action = get_action(env)
        new_obs, reward, terminated, truncated, new_info = env.step(action)
        
        obs = obs[:186, :160, :]        # First crop to (186, 160) to only keep the game area in focus (not the score or the remaining lives)
        obs = cv2.resize(obs, [64, 64], interpolation=cv2.INTER_AREA)   # Then do downsampling / stretching to (64, 64)
        
        obs_list.append(obs / 255.0)
        act_list.append(action)
        reward_list.append(float(reward))
        lives_list.append(info["lives"])
        
        obs = new_obs
        info = new_info
        
        if (terminated or truncated): 
            break

    save_data(idx, obs_list, act_list, reward_list, lives_list)
    env.close()
    
    return idx

# Multiprocessing
if __name__ == "__main__":
    import multiprocessing as mp
    N_WORKERS = 8

    with mp.Pool(processes=N_WORKERS) as pool:
        results = pool.map(generate_one_episode, range(NUM_ROLLOUTS))

    print(f"Generated {len(results)} episodes.")
    print(f"Files saved to {SAVE_PATH.resolve()}")
