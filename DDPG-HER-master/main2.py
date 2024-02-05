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
import sys
import tracemalloc


ENV_NAME = 'SafetyRacecarGoal0-v0'
RENDER_MODE = 'human'
# RENDER_MODE = None
INTRO = False
Train = True
Play_FLAG = False
MAX_EPOCHS = 50
MAX_CYCLES = 50
num_updates = 40
MAX_EPISODES = 2
memory_size = 7e+5 // 50
batch_size = 256
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.98
tau = 0.05
k_future = 4

test_env = safety_gymnasium.make(ENV_NAME,render_mode = RENDER_MODE)
#state_shape = test_env.observation_space.shape
state_shape = (30,0)
n_actions = test_env.action_space.shape[0]
#n_goals = test_env.observation_space["goal"].shape[0]
n_goals = 2
action_bounds = [test_env.action_space.low, test_env.action_space.high]
to_gb = lambda in_bytes: in_bytes / 1024 / 1024 / 1024


def eval_agent(env_, agent_):
    print("eval_agent")
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
        while np.linalg.norm(ag - g) <= 0.05:
            env_dictionary, info = env_.reset()
            s = env_dictionary[:-2] #observation
            ag = env_dictionary[:2] #achieved goal
            g = env_dictionary[-2:] #desired goal
        ep_r = 0
        i = 0
        success_score = 0
        while not (terminated or truncated):
            with torch.no_grad():
                a = agent_.choose_action(s, g, train_mode=False)
            observation_new, r, c, terminated, truncated, info_ = env_.step(a)
            s = observation_new[:-2] #observation
            g = observation_new[-2:] #desired goal
            if np.linalg.norm(s[:2] - g) < 0.5:
                success_score += 1
            # per_success_rate.append(info_['is_success'])
            ep_r += r
            i += 1 
        total_success_rate.append( success_score/i )
        if ep == 0:
            running_r.append(ep_r)
        else:
            running_r.append(running_r[-1] * 0.99 + 0.01 * ep_r)
    total_success_rate = np.array(total_success_rate)
    local_success_rate = np.mean(total_success_rate[:])
    return local_success_rate, running_r, ep_r


if INTRO:
    print(f"state_shape:{state_shape[0]}\n"
          f"number of actions:{n_actions}\n"
          f"action boundaries:{action_bounds}\n"
          f"max timesteps:{test_env._max_episode_steps}")
    for _ in range(3):
        done = False
        terminated = False
        truncated = False
        test_env.reset()
        while not (terminated or truncated):
            action = test_env.action_space.sample()
            # print("action",action)
            test_observation, test_reward, test_cost, terminated, truncated, test_info = test_env.step(action)
            # observation_new, r, c, terminated, truncated, info_ 
            # substitute_goal = test_state["achieved_goal"].copy()
            # substitute_reward = test_env.compute_reward(
            #     test_state["achieved_goal"], substitute_goal, test_info)
            # print("r is {}, substitute_reward is {}".format(r, substitute_reward))
            #test_env.render()
        test_env.close()
    exit(0)

env = safety_gymnasium.make(ENV_NAME)

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

eval_agent(env, agent)  


if Train:

    tracemalloc.start() #1

    t_success_rate = []
    total_ac_loss = []
    total_cr_loss = []
    for epoch in range(MAX_EPOCHS): #50
        start_time = time.time()
        epoch_actor_loss = 0
        epoch_critic_loss = 0
        for cycle in range(0, MAX_CYCLES): #50
            mb = []
            cycle_actor_loss = 0
            cycle_critic_loss = 0
            for episode in range(MAX_EPISODES): #2
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
                
                ####
                env_dict,info = env.reset()
                state = env_dict[:-2] #observation
                achieved_goal = env_dict[:2] #achieved goal
                desired_goal = env_dict[-2:] #desired goal
                #initialize achieved_goal and desired_goal
                while np.linalg.norm(achieved_goal - desired_goal) <= 0.1:
                    env_dict,info = env.reset()
                    state = env_dict[:-2] #observation
                    achieved_goal = env_dict[:2] #achieved goal
                    desired_goal = env_dict[-2:] #desired goal
                while not (terminated or truncated):
                    action = agent.choose_action(state, desired_goal)

                    ###
                    next_env_dict, reward, cost,terminated, truncated, info = env.step(action)

                    next_state = next_env_dict[:-2] #observation
                    next_achieved_goal = next_env_dict[:2] #achieved goal
                    next_desired_goal = next_env_dict[-2:] #desired goal

                    episode_dict["state"].append(state.copy())
                    episode_dict["action"].append(action.copy())
                    episode_dict["achieved_goal"].append(next_achieved_goal.copy())
                    episode_dict["desired_goal"].append(next_desired_goal.copy())

                    state = next_state.copy()
                    achieved_goal = next_achieved_goal.copy() # I don't know why it is here.
                    desired_goal = next_desired_goal.copy()
                    
                # env.close()

                episode_dict["state"].append(state.copy())
                episode_dict["achieved_goal"].append(achieved_goal.copy())
                episode_dict["desired_goal"].append(desired_goal.copy())
                episode_dict["next_state"] = episode_dict["state"][1:]
                episode_dict["next_achieved_goal"] = episode_dict["achieved_goal"][1:]
                mb.append(dc(episode_dict))
            
            agent.store(mb)
            for n_update in range(num_updates): #40
                actor_loss, critic_loss = agent.train()
                cycle_actor_loss += actor_loss
                cycle_critic_loss += critic_loss

            epoch_actor_loss += cycle_actor_loss / num_updates
            epoch_critic_loss += cycle_critic_loss /num_updates
            agent.update_networks()


        ram = psutil.virtual_memory()
        success_rate, running_reward, episode_reward = eval_agent(env, agent)
        total_ac_loss.append(epoch_actor_loss)
        total_cr_loss.append(epoch_critic_loss)
        
        t_success_rate.append(success_rate)
    
        snapshot = tracemalloc.take_snapshot()  #2
        top_stats = snapshot.statistics('lineno')

        print("[ Top 10 ]")
        for stat in top_stats[:10]:
            print(stat)
        print("---")
        print(f"Epoch:{epoch}| "
                f"Running_reward:{running_reward[-1]:.3f}| "
                f"EP_reward:{episode_reward:.3f}| "
                f"Memory_length:{len(agent.memory)}| "
                f"Duration:{time.time() - start_time:.3f}| "
                f"Actor_Loss:{actor_loss:.3f}| "
                f"Critic_Loss:{critic_loss:.3f}| "
                f"Success rate:{success_rate:.3f}| "
                f"{to_gb(ram.used):.1f}/{to_gb(ram.total):.1f} GB RAM|"
                f"{to_gb(ram.active):.1f},{to_gb(ram.inactive):.1f}GB RAM|"
                f"buffers {ram.buffers}|"
                f"memory length:{len(agent.memory)}/{agent.capacity}, {sys.getsizeof(agent.memory.memory)} byte")
        agent.save_weights()



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
    # player.evaluate()
    player.play_only_once()
