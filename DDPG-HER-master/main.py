import safety_gymnasium
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from play import Play
# import mujoco_py
import random
from mpi4py import MPI
import psutil
import time
from copy import deepcopy as dc
import os
import torch

ENV_NAME = 'SafetyRacecarGoal0-v0'
RENDER_MODE = 'human' #'human’, ‘rgb_array’, ‘depth_array’.
INTRO = False
Train = True
Play_FLAG = False
MAX_EPOCHS = 2000
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 2
memory_size = 7e+5 // 50
batch_size = 256
actor_lr = 1e-2 # 1e-3
critic_lr = 1e-2 # 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = safety_gymnasium.make(ENV_NAME,render_mode = RENDER_MODE)
#state_shape = test_env.observation_space.shape
state_shape = (20,0)
n_actions = test_env.action_space.shape[0]
#n_goals = test_env.observation_space["goal"].shape[0]
n_goals = 2
action_bounds = [test_env.action_space.low, test_env.action_space.high]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['IN_MPI'] = '1'


def eval_agent(env_, agent_):
    total_success_rate = []
    running_r = []
    for ep in range(10):
        per_success_rate = []
        terminated = False
        truncated = False
        env_dictionary, info = env_.reset()
        s = env_dictionary[:-2] #observation
        ag = env_dictionary[:2] #achieved goal
        g = env_dictionary[-2:] #desired goal
        while np.linalg.norm(ag - g) <= 0.3:
            env_dictionary = env_.reset()
            s = env_dictionary[:-2]
            ag = env_dictionary[:2]
            g = env_dictionary[-2:]
        ep_r = 0
        while not (terminated or truncated):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, c, terminated, truncated, info_ = env_.step(a)
            s = observation_new[:-2]
            g = observation_new[-2:]
            if np.linalg.norm(s[:2] - g) < 0.3:
                per_success_rate.append(True)
            else:
                per_success_rate.append(False)
            ep_r += r
        total_success_rate.append(per_success_rate)
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:, -1])
    global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
    return global_success_rate / MPI.COMM_WORLD.Get_size(), running_r, ep_r


if INTRO:
    print(f"state_shape:{state_shape[0]}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        test_env.reset()
        while not done:
            action = test_env.action_space.sample()
            test_state, test_reward, test_done, test_info = test_env.step(action)
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            test_env.render()
    exit(0)

env = safety_gymnasium.make(ENV_NAME, render_mode = RENDER_MODE) # render_mode 'human' 'rgb_array', 'depth_array'
# env.set_seed(MPI.COMM_WORLD.Get_rank())
# random.seed(MPI.COMM_WORLD.Get_rank())
# np.random.seed(MPI.COMM_WORLD.Get_rank())
# torch.manual_seed(MPI.COMM_WORLD.Get_rank())
agent = Agent(n_states=state_shape,
              n_actions=n_actions,
              n_goals=n_goals,
              action_bounds=action_bounds,
              capacity=memory_size,
              action_size=n_actions,
              batch_size=batch_size,
              actor_lr=actor_lr,
              critic_lr=critic_lr,
              gamma=gamma,
              tau=tau,
              k_future=k_future,
              env=dc(env))

# eval_agent(env,agent)
# print("start")

if Train:

    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS):
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES):
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES):
                episode_dict = {
                    "state": [],
                    "action": [],
                    "info": [],
                    "achieved_goal": [],
                    "desired_goal": [],
                    "next_state": [],
                    "next_achieved_goal": []}
                terminated = False
                truncated = False
                env_dict,info = env.reset()
                state = env_dict[:-2] #observation
                achieved_goal = env_dict[:2] #achieved goal
                desired_goal = env_dict[-2:] #desired goal
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.3:
                    env_dict,info = env.reset()
                    state = env_dict[:-2] #observation
                    achieved_goal = env_dict[:2] #achieved goal
                    desired_goal = env_dict[-2:] #desired goal
                # for t in range(50):
                while not (terminated or truncated):
                    action = agent.choose_action(state, desired_goal)
                    next_env_dict, reward, cost,terminated, truncated, info = env.step(action)

                    next_state = next_env_dict[:-2] #observation
                    next_achieved_goal = next_env_dict[:2] #achieved goal
                    next_desired_goal = next_env_dict[-2:] #desired goal

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(achieved_goal.copy())
                    episode_dict["desired_goal"].append(desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy()
                    desired_goal = next_desired_goal.copy()
                    # env.render()
                # env.close()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))
            agent.store(mb)
            for n_update in range(num_updates):
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()
            # print("update network")

        ram = psutil.virtual_memory()
        success_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        if MPI.COMM_WORLD.Get_rank() == 0:
            t_success_rate.append(success_rate)
            print(f"Epoch:{epoch}| "
                  f"Running_reward:{running_reward[-1]:.3f}| "
                  f"EP_reward:{episode_reward:.3f}| "
                  f"Memory_length:{len(agent.memory)}| "
                  f"Duration:{time.time() - start_time:.3f}| "
                  f"Actor_Loss:{actor_loss:.3f}| "
                  f"Critic_Loss:{critic_loss:.3f}| "
                  f"Success rate:{success_rate:.3f}| "
                  f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM")
            agent.save_weights()

    if MPI.COMM_WORLD.Get_rank() == 0:

        with SummaryWriter("logs") as writer:
            for i, success_rate in enumerate(t_success_rate):
                writer.add_scalar("Success_rate", success_rate, i)

        plt.style.use('ggplot')
        plt.figure()
        plt.plot(np.arange(0, MAX_EPOCHS), t_success_rate)
        plt.title("Success rate")
        plt.savefig("success_rate.png")
        plt.show()

elif Play_FLAG:
    player = Play(env, agent, max_episode=100)
    player.evaluate()
