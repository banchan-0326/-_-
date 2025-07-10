import os
import numpy
import gym
import gym_building # Needs to be imported to register the env
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy # LnMlpPolicy is for older versions, MlpPolicy is common. For specific layer norm, use CnnLnLstmPolicy or import specific policy
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines import DQN

# Assuming 'setting.py' is in the parent directory 'model/'
# To make this runnable from Smart-Elevator-RL-Clone directory:
# Option 1: Add Smart-Elevator-RL-Clone to PYTHONPATH
# Option 2: Modify sys.path (less ideal for packaged code but okay for scripts)
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../..')) # Add project root

from model.setting import people, NUM_LIFTS, HEIGHT_OF_BUILDING, GYM_ENV_NAME, MODEL_SAVE_PATH
from model.setting import LEARNING_RATE, DISCOUNT_FACTOR, EPSILON, EPSILON_DECAY, EPSILON_MIN
from model.setting import BATCH_SIZE, TRAIN_START, MEMORY_SIZE, EPISODES, MAX_STEPS_PER_EPISODE

# Training Hyperparameters from setting.py
# TIMESTEPS = 1000000 # Total number of training steps

# For logging and model saving
LOG_DIR_BASE = os.path.join(os.path.dirname(__file__), "dqn_log")
MODEL_DIR_BASE = os.path.join(os.path.dirname(__file__), "dqn_models")
os.makedirs(LOG_DIR_BASE, exist_ok=True)
os.makedirs(MODEL_DIR_BASE, exist_ok=True)

# Unique log and model directories for this run
# run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# log_directory = os.path.join(LOG_DIR_BASE, run_id)
# model_directory = os.path.join(MODEL_DIR_BASE, run_id)
# os.makedirs(log_directory, exist_ok=True)
# os.makedirs(model_directory, exist_ok=True)

# Using fixed names for simplicity as in original
log_directory = LOG_DIR_BASE
model_directory = MODEL_DIR_BASE


best_mean_reward = -numpy.inf
n_steps_callback = 0 # Renamed to avoid conflict with _locals

def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global best_mean_reward, n_steps_callback
    # Logs will be saved in log_directory by the Monitor wrapper
    if (n_steps_callback + 1) % 20000 == 0: # Check every 20000 steps
        # stable-baselines Monitor already saves results, load_results expects a path to that dir
        x, y = ts2xy(load_results(log_directory), 'timesteps')
        if len(x) > 0:
            mean_reward = numpy.mean(y[-100:]) # Mean reward over last 100 episodes
            print(f"Timesteps: {x[-1]}")
            print(f"Best mean reward: {best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                print("Saving new best model")
                # Save the model
                model_path = os.path.join(model_directory, f'dqn_model_{n_steps_callback + 1}.pkl')
                _locals['self'].save(model_path)
                print(f"Model saved to {model_path}")
    n_steps_callback += 1
    return True


if __name__ == "__main__":
    print(f"Initializing environment: {GYM_ENV_NAME}")
    print(f"Number of people: {len(people)}")
    print(f"Lifts: {NUM_LIFTS}, Building Height: {HEIGHT_OF_BUILDING}")

    # Create and wrap the environment
    env = gym.make(GYM_ENV_NAME, people=people, num_of_lift=NUM_LIFTS, height_of_building=HEIGHT_OF_BUILDING)
    env = Monitor(env, log_directory, allow_early_resets=True)
    env = DummyVecEnv([lambda: env]) # Vectorized environment

    # DQN model parameters from settings or defaults
    # For stable-baselines DQN, some params are passed directly, others are part of policy_kwargs or other args.
    # Default MlpPolicy is (64,64)
    # buffer_size is MEMORY_SIZE
    # learning_starts is TRAIN_START
    # batch_size is BATCH_SIZE
    # gamma is DISCOUNT_FACTOR
    # exploration_fraction, exploration_final_eps are related to epsilon decay.
    # learning_rate is LEARNING_RATE

    model = DQN(
        env=env,
        policy=MlpPolicy, # Using MlpPolicy, LnMlpPolicy might be specific or older.
        verbose=1,
        learning_rate=LEARNING_RATE,
        buffer_size=MEMORY_SIZE,        # Replay buffer size
        learning_starts=TRAIN_START,    # How many steps of random actions before learning starts
        batch_size=BATCH_SIZE,
        gamma=DISCOUNT_FACTOR,          # Discount factor
        exploration_fraction=0.1,       # Fraction of entire training period over which epsilon is annealed
        exploration_final_eps=EPSILON_MIN, # Final value of epsilon
        # train_freq=4,                 # Update the model every 4 steps
        # gradient_steps=1,             # How many gradient steps to do after each rollout
        tensorboard_log=os.path.join(log_directory, "dqn_tensorboard/"), # Path for tensorboard logs
        # policy_kwargs=dict(layers=[64, 64]) # Example: Customize network architecture
    )

    # To load a pre-trained model:
    # model_load_path = os.path.join(model_directory, "dqn_model_XXXX.pkl") # Specify model name
    # if os.path.exists(model_load_path):
    #     print(f"Loading model from {model_load_path}")
    #     model = DQN.load(model_load_path, env=env)
    # else:
    #     print("No pre-trained model found, starting from scratch.")

    total_timesteps = EPISODES * MAX_STEPS_PER_EPISODE # Calculate total timesteps
    print(f"Starting training for {total_timesteps} timesteps...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback
        )
    except Exception as e:
        print(f"An error occurred during training: {e}")
    finally:
        # Save the final model
        final_model_path = os.path.join(model_directory, 'dqn_model_final.pkl')
        model.save(final_model_path)
        print(f"Training finished. Final model saved to {final_model_path}")
        env.close()
