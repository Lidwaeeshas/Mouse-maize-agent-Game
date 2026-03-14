import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2 as cv
from google.colab.patches import cv2_imshow



class GridEnv(gym.Env):
  def __init__(self,size = 6):
    self.size = size
    self.action_space = spaces.Discrete(4) #sets the total number of valid actions

    #this rule variabe set the only boundry the agent is allowed to
    self.observation_space = spaces.Dict({
    "agent" : spaces.Box(low = np.array([0,0]), high = np.array([size -1,size -1]), dtype = np.int32),
    "target" : spaces.Box(low = np.array([0,0]), high = np.array([size -1,size -1]), dtype = np.int32)
    })

    #initial position if the agent
    self.agent_pos = np.array([0,0])

    #target position
    self.target = np.array([5,4])

    #obstacles
    self.obs = [np.array([0,2]),np.array([0,5]),np.array([4,0]),np.array([1,2]),np.array([2,0]),np.array([3,2]),np.array([5,5]),np.array([4,3]),np.array([2,4]),np.array([3,1]),np.array([5,3]),np.array([5,1]),np.array([2,5]),np.array([4,0])]

    self.action_maping = {
    0: np.array([-1,0]),
    1: np.array([1,0]),
    2: np.array([0,-1]),
    3: np.array([0,1])
    }

  def _get_obs(self):
    return {"agent": self.agent_pos.copy(),"target": self.target.copy()}


  def _new_agent_pos(self,compares):
    for ob in compares:
      pos = np.random.randint(0,6,size = 2)
      if not any(np.array_equal(pos,ob)for ob in obs):
        return pos
    return 0


  def _new_target_pos(self,agent,obs):
    for ob in obs:
      proposal = np.random.randint(0,6,size = 2)
      if not any(np.array_equal(proposal,ob) for ob in obs) and not np.array_equal(agent,obs):
        return proposal
    return None

  def reset(self,seed = None):
    super().reset(seed = seed)
    self.agent_pos = self._new_agent_pos(self.obs)
    self.target = self._new_target_pos(self.agent_pos,self.obs)
    return self._get_obs(),{}

  def step(self,action):
    action = int(action)
    new_pos = self.agent_pos + self.action_maping[action]

    self.agent_pos = np.clip(new_pos,0,self.size -1)

    reward = -0.01
    terminated = False
    truncated = False

    reward -= 1 if any(np.array_equal(self.agent_pos,obs) for obs in self.obs) else -0.01

    if np.array_equal(self.agent_pos,self.target):
     reward += 10
     terminated = True

    return self._get_obs(),reward,terminated,truncated,{}

  def render(self):
    window_size = 400
    size = 6
    box_s = window_size//size
    img = np.ones((window_size,window_size,3),dtype = np.uint8)*255
    for i in range(size +1):
      pos = i * box_s
      cv.line(img,(0,pos),(window_size,pos),(0,0,0),2)
      cv.line(img,(pos,0),(pos,window_size),(0,0,0),2)
    x, y = self.agent_pos[0],self.agent_pos[1]
    x,y = ((x * box_s) +(box_s//2)),((y * box_s) +(box_s//2))
    center = (x,y)
    cv.circle(img,center,3,(255,0,0),4)
    for obs in self.obs:
      x,y = (obs[0]*box_s) + box_s//2 -10, (obs[1]*box_s) + box_s//2 -10
      center = (x,y)
      cv.circle(img,center,10,(0,0,255),-1)
    x,y = (self.target[0]*box_s) + box_s//2 -10, (self.target[1]*box_s) + box_s//2 -10
    center = (x,y)
    cv.circle(img,center,10,(0,255,0),-1)
    return img



