# import the objects you want to use
# or you can define specific objects by yourself, just make sure obeying our specification
from safety_gymnasium.assets.geoms import Apples
from safety_gymnasium.bases import BaseTask

# inherit the basetask
class MytaskLevel0(BaseTask):
    def __init__(self, config):
        super().__init__(config=config)
        # define some properties
        self.num_steps = 500
        self.agent.placements = [(-0.8, -0.8, 0.8, 0.8)]
        self.agent.keepout = 0
        self.lidar_conf.max_dist = 6
        # add objects into environments
        self.add_geoms(Apples(num=2, size=0.3))

    def calculate_reward(self):
        # implement your reward function
        # Note: cost calculation is based on objects, so it's automatic
        reward = 0.0
        dist_goal = self.dist_goal()
        reward += (self.last_dist_goal - dist_goal) * self.goal.reward_distance
        self.last_dist_goal = dist_goal

        if self.goal_achieved:
            reward += self.goal.reward_goal

        return reward

    def specific_reset(self):
        # depending on your task
        pass

    def specific_step(self):
        # depending on your task
        pass

    def update_world(self):
        # depending on your task
        self.build_goal_position()
        self.last_dist_goal = self.dist_goal()

    @property
    def goal_achieved(self):
        # depending on your task
        return self.dist_goal() <= self.goal.size