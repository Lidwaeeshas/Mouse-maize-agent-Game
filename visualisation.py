from IPython.display import clear_output
import time

env = GridEnv()
obs,info = env.reset()

model = PPO.load("/content/my_mouse.zip")

for _ in range(100):
  action,_ = model.predict(obs)
  obs,rew,ter,tru,info = env.step(action)
  if ter:
    obs,info = env.reset()
  clear_output(wait = True)
  frame = env.render()

  cv2_imshow(frame)
  time.sleep(2)
