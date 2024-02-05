import safety_gymnasium
import numpy as np
env_id = 'SafetyRacecarGoal0-v0'
env = safety_gymnasium.make(env_id,render_mode = 'human')


for i in range(10):
    obs, info = env.reset()
    # print(env.observation_space)
    # print(env.obs_space_dict)
    start_location = obs[:2]
    # print("start_location",obs[:2])

    terminated = False
    truncated = False
    while not (terminated or truncated):
        # act = env.action_space.sample()
        # print(act)
        act = [2,0]
        
        obs, reward, cost, terminated, truncated, info = env.step(act)
        if cost !=0 :
            print(cost)
        # print(cost)
        # if i%30==0:
            # print(obs)
        env.render()
    # print("last_location",obs[:2])
    # print("distance",np.linalg.norm(start_location - obs[:2]))
            
        
