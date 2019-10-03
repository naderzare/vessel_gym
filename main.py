# from car import *
# env = MountainCarEnv()
2
from vessel_gym import *
env = VesselEnv()

state = env.reset()

done = False
while not done:
    env.render()
    state, reward, done, s = env.step(int(input('enter action:')))

env.close()

