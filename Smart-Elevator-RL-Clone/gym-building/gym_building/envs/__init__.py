from gym_building.envs.Building import BuildingEnv
from gym.envs.registration import register

register(
    id='BuildingEnv-v0',
    entry_point='gym_building.envs:BuildingEnv',
)
