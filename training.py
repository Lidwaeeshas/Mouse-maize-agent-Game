from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

#creating the environment 
env = GridEnv()

#we use MultiInputPolicy because it is the best algorithm for non flattened vectors
model = PPO(
"MultiInputPolicy",
env = env,
verbose = 1,
gamma = 0.99,
n_steps = 2048,
batch_size= 32,
n_epochs = 12
)

model.learn(total_timesteps = 6e5)

model.save("my_mouse")
