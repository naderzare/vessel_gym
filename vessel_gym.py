import math
import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


def calc_dist(a, b):
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

        self.vessel_position_x = 0.
        self.vessel_position_y = 0.
        self.vessel_head_angle = 0.
        self.vessel_vel_v = 0.
        self.vessel_vel_u = 0.
        self.vessel_vel_r = 0.
        self.vessel_radius = 0.1
        self.state = np.array([self.vessel_position_x,self.vessel_position_y, self.vessel_head_angle, self.vessel_vel_u, self.vessel_vel_v, self.vessel_vel_r])
        self.goal_position = np.array([2.5, 2.5])
        self.goal_radius = 0.1
        self.obstacles_number = 9
        self.obstacles_position = [[1.0, 0.5], [0.5, 1.0], [0.75, 1.5], [0.25, 2.25], [1.0, 2.0], [1.5, 1.5],
                                   [2.0, 0.25], [2.0, 1.0], [2.5, 0]]
        self.obstacles_radius = 0.1
        self.max_speed = 0.5

        self.viewer = None
        self.action_space = spaces.Box(low=-1, high=+1,
                                       shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-10.0, -10.0]), high=np.array([10.0, 10.0]), dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def R(self):
        res = np.zeros((3,3))
        res[0][0] = np.cos(self.vessel_head_angle)
        res[1][1] = np.cos(self.vessel_head_angle)
        res[0][1] = -np.sin(self.vessel_head_angle)
        res[1][0] = np.sin(self.vessel_head_angle)
        res[2][2] = 1
        return res

    def step(self, action):
        print(action)
        dV = np.array([action[0], 0, action[1]])
        print('dV:', dV)
        print('R:', self.R())
        dP = np.dot(self.R(), dV)
        print('dp', dP)
        self.vessel_position_x += dP[0]
        self.vessel_position_y += dP[1]
        self.vessel_head_angle += dP[2]

        done = False
        reward = -1.0
        if self.vessel_position_x < self.min_x or self.vessel_position_x > self.max_x or self.vessel_position_y < self.min_y or self.vessel_position_y > self.max_y:
            done = True
            reward = -1.0
        elif self.is_collision_obstacle():
            done = True
            reward = -1.0
        elif self.is_collision_goal():
            done = True
            reward = 0.0

        self.state = np.array([self.vessel_position_x,self.vessel_position_y, self.vessel_head_angle, self.vessel_vel_u, self.vessel_vel_v, self.vessel_vel_r])
        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = np.array([self.vessel_position_x,self.vessel_position_y, self.vessel_head_angle, self.vessel_vel_u, self.vessel_vel_v, self.vessel_vel_r])
        return self.state

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 600

        world_width = self.max_x - self.min_y
        scale = screen_width/world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            xs = [0, 2, 3, 2, 0]
            ys = [1, 1, 0, -1, -1]
            for i in range(5):
                xs[i] *= 10
                ys[i] *= 10
            print(xs)
            xys = list(zip(xs, ys))
            vessel_circle = rendering.make_polygon(xys)
            # vessel.add_attr(rendering.Transform(translation=(0, 0)))
            vessel_circle.set_color(.0, .0, 1)
            self.cartrans = rendering.Transform()
            vessel_circle.add_attr(self.cartrans)
            self.viewer.add_geom(vessel_circle)

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

        pos = [self.vessel_position_x, self.vessel_position_y]
        self.cartrans.set_translation((pos[0] - self.min_x) * scale, (pos[1] - self.min_y) * scale)
        self.cartrans.set_rotation(self.vessel_head_angle)
        self.target_c.set_translation((2.5 - self.min_x) * scale, (2.5 - self.min_y) * scale)
        for c, p in self.obstacles_cp:
            c.set_translation(p[0], p[1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def is_collision_obstacle(self):
        collision = lambda obstacle_pos: True if calc_dist((self.vessel_position_x,self.vessel_position_y),
                                                           obstacle_pos) < self.obstacles_radius + self.vessel_radius else False
        if any(list(map(collision, self.obstacles_position))):
            return True
        return False

    def is_collision_goal(self):
        if calc_dist((self.vessel_position_x,self.vessel_position_y), self.goal_position) < self.goal_radius + self.vessel_radius:
            return True
        return False

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
