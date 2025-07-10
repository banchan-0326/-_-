# This file makes 'gym_building' a Python package.
# It can also be used for package-level imports or initializations if needed.

# For example, you could import key classes here to make them available as:
# from gym_building import BuildingEnv
# However, the current structure relies on gym.make() using the entry_point
# specified in envs/__init__.py, so this file can remain empty or minimal.

# Ensure the custom environment is registered when this package is imported at a higher level,
# though the primary registration happens in gym_building.envs.__init__
# and is triggered by `import gym_building` in train.py
import gym_building.envs
