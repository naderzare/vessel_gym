# from car import *
# env = MountainCarEnv()
import time
from vessel_gym import *
env = VesselEnv()

state = env.reset()

done = False

acts = [(0.2,0.0),(0.2,1),(0.2,0.5),(0.2,-0.5),(0.2,-0.3),(0.2,-0.2),(0.2,0.5),(0.2,0.4),(0.2,0.3),(0.2,0.3),(0.2,-0.3),(0.2,-0.3),(0.2,0.3)]
acs = 0
while not done:
    env.render()
    # ac = input()
    # u, r = ac.split(',')
    # u = float(u)
    # r = float(r)
    time.sleep(0.5)
    state, reward, done, s = env.step(acts[acs])
    acs+=1

env.close()

