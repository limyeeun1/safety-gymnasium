import torch
from torch import device
import numpy as np
#import cv2
from gym import wrappers
from mujoco_py import GlfwContext
import safety_gymnasium


GlfwContext(offscreen=True)

from mujoco_py.generated import const


class Play:
    def __init__(self, env, agent, max_episode=4):
        self.env = env
        # self.env = wrappers.Monitor(env, "./videos", video_callable=lambda episode_id: True, force=True)
        self.max_episode = max_episode
        self.agent = agent
        self.agent.load_weights()
        self.agent.set_to_eval_mode()
        self.device = device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self):

        for _ in range(self.max_episode):
            env_dict, info = self.env.reset()
            state = env_dict[:-2]
            achieved_goal = env_dict[:2]
            desired_goal = env_dict[-2:]
            while np.linalg.norm(achieved_goal - desired_goal) <= 0.1:
                env_dict = self.env.reset()
                state = env_dict[:-2]
                achieved_goal = env_dict[:2]
                desired_goal = env_dict[-2:]
            # done = False
            terminated , truncated = False, False
            episode_reward = 0
            while not (terminated or truncated):
                action = self.agent.choose_action(state, desired_goal, train_mode=False)
                # print("action",action)
                # next_env_dict, r, done, _ = self.env.step(action)
                next_env_dict, r, cost, terminated, truncated, info = self.env.step(action)
                next_state = next_env_dict[:-2]
                next_desired_goal = next_env_dict[-2:]
                episode_reward += r
                state = next_state.copy()
                desired_goal = next_desired_goal.copy()
                I = self.env.render()  # mode = "rgb_array
                # self.env.viewer.cam.type = const.CAMERA_FREE
                # self.env.viewer.cam.fixedcamid = 0
                # I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                # cv2.imshow("I", I)
                # cv2.waitKey(2)
            print(f"episode_reward:{episode_reward:3.3f}")

        self.env.close()
    
    def play_only_once(self):

        for _ in range(self.max_episode):
            env_dict, info = self.env.reset()
            state = env_dict[:-2]
            achieved_goal = env_dict[:2]
            desired_goal = env_dict[-2:]

            while np.linalg.norm(achieved_goal - desired_goal) <= 0.1:
                env_dict = self.env.reset()
                state = env_dict[:-2]
                achieved_goal = env_dict[:2]
                desired_goal = env_dict[-2:]
            # print("start")
            terminated = False
            truncated = False
            epi_reward = 0
            while not (terminated or truncated) :
                action = self.agent.choose_action(state, desired_goal, train_mode=False)
                # print(action)
                # action = [1,1]
                obs, reward, cost, terminated, truncated, info = self.env.step(action)
                state = obs[:-2]
                epi_reward += reward
                self.env.render()
            print(f"episode_reward:{epi_reward:3.3f}")
            # print("epi end")

