import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


def calc_dist(a, b):
    print(a, b)
    dist = math.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
    return dist


class VesselEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.min_x = -0.5
        self.min_y = -0.5
        self.max_x = 3
        self.max_y = 3

        self.vessel_position = np.array([0., 0.])
        self.vessel_radius = 0.1
        self.state = self.vessel_position
        self.goal_position = np.array([2.5, 2.5])
        self.goal_radius = 0.1
        self.obstacles_number = 9
        self.obstacles_position = [[1.0, 0.5], [0.5, 1.0], [0.75, 1.5], [0.25, 2.25], [1.0, 2.0], [1.5, 1.5],
                                   [2.0, 0.25], [2.0, 1.0], [2.5, 0]]
        self.obstacles_radius = 0.1
        self.max_speed = 0.5

        self.viewer = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        position = self.state
        action2velocity = [np.array([self.max_speed, 0.0]),np.array([0.0, self.max_speed]),np.array([-self.max_speed, 0.0]),np.array([0.0, -self.max_speed])]
        position += action2velocity[action]

        done = False
        reward = -1.0
        if position[0] < self.min_x or position[0] > self.max_x or position[1] < self.min_y or position[1] > self.max_y:
            done = True
            reward = -1.0
        elif self.is_collision_obstacle():
            done = True
            reward = -1.0
        elif self.is_collision_goal():
            done = True
            reward = 0.0

        self.state = position
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.vessel_position
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.max_x - self.min_y
        scale = screen_width/world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            vessel = rendering.make_circle(self.vessel_radius * scale)
            # vessel.add_attr(rendering.Transform(translation=(0, 0)))
            vessel.set_color(.0, .0, 1)
            self.cartrans = rendering.Transform()
            vessel.add_attr(self.cartrans)
            self.viewer.add_geom(vessel)

            obstacles = [rendering.make_circle(self.obstacles_radius * scale) for _ in range(self.obstacles_number)]
            self.obstacles_cp = []
            for o, pos in zip(obstacles, self.obstacles_position):
                o.set_color(1, 0, 0)
                x = (pos[0] - self.min_x) * scale
                y = (pos[1] - self.min_y) * scale
                # o.add_attr(rendering.Transform(translation=(x, y)))
                self.obstacles_cp.append([rendering.Transform(),[x, y]])
                o.add_attr(self.obstacles_cp[-1][0])
                self.viewer.add_geom(o)

            target = rendering.make_circle(self.goal_radius * scale)
            target.set_color(.0, 1, 0)
            self.target_c = rendering.Transform()
            target.add_attr(rendering.Transform(translation=(0, 0)))
            target.add_attr(self.target_c)
            self.viewer.add_geom(target)

        pos = self.state
        self.cartrans.set_translation((pos[0] - self.min_x) * scale, (pos[1] - self.min_y) * scale)
        self.target_c.set_translation((2.5 - self.min_x) * scale, (2.5 - self.min_y) * scale)
        for c, p in self.obstacles_cp:
            c.set_translation(p[0], p[1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def is_collision_obstacle(self):
        collision = lambda obstacle_pos: True if calc_dist(self.vessel_position,
                                                           obstacle_pos) < self.obstacles_radius + self.vessel_radius else False
        if any(list(map(collision, self.obstacles_position))):
            return True
        return False

    def is_collision_goal(self):
        if calc_dist(self.vessel_position, self.goal_position) < self.goal_radius + self.vessel_radius:
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
