from gym_building.envs.Mission import Mission
from gym_building.envs.Person import Person

# Initialize people: 25 people working on 4th floor, 25 people working on 5th floor
# Assuming building height will accommodate this. Min/max height in Person refers to their default working floor.
# If the building is, for example, 5 floors high, then Person(4,4) means they work on the 4th floor.
# Person(min_height, max_height) from the original code seems to imply they pick a random default floor
# within this range. For this setting, it's fixed.

people = []
num_people_floor_4 = 25
num_people_floor_5 = 25

# Create people for 4th floor
for _ in range(num_people_floor_4):
    # In Person class, default_layer is randomly chosen between min_height and max_height (inclusive).
    # So, to ensure they are on 4th floor, min_height and max_height should be 4.
    people.append(Person(min_height=4, max_height=4))

# Create people for 5th floor
for _ in range(num_people_floor_5):
    people.append(Person(min_height=5, max_height=5))


# Define Missions
# Times are in seconds from midnight (00:00)
breakfast = Mission(target=2, start_time=7.5 * 60 * 60, end_time=8.5 * 60 * 60) # Go to 2nd floor for breakfast
morning_conference = Mission(target=3, start_time=9 * 60 * 60, end_time=10 * 60 * 60) # Go to 3rd floor for morning conference
lunch = Mission(target=2, start_time=13 * 60 * 60, end_time=14 * 60 * 60) # Go to 2nd floor for lunch
dinner_conference = Mission(target=3, start_time=16 * 60 * 60, end_time=17 * 60 * 60) # Go to 3rd floor for dinner conference
off_work = Mission(target=1, start_time=18 * 60 * 60, end_time=23 * 60 * 60 + 59 * 60 + 59) # Go to 1st floor to go home (until 23:59:59)

# Assign missions to people
for person in people:
    person.add_mission(breakfast)
    person.add_mission(lunch)
    person.add_mission(off_work)
    if person.default_layer == 4: # Only 4th-floor workers attend morning conference
            person.add_mission(morning_conference)
    if person.default_layer == 5: # Only 5th-floor workers attend dinner conference
            person.add_mission(dinner_conference)

# Building specific settings (can be overridden in train.py if needed)
NUM_LIFTS = 2
HEIGHT_OF_BUILDING = 5 # Assuming a 5-story building based on Person assignments.
                        # The Person class expects min_height, max_height for random default floor assignment.
                        # Here, default_layer is fixed to 4 or 5.
                        # Make sure HEIGHT_OF_BUILDING is consistent with these floors.
GYM_ENV_NAME = 'BuildingEnv-v0'

# DQN Agent settings
STATE_SIZE = None # To be determined by the environment's observation space
ACTION_SIZE = None # To be determined by the environment's action space
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.01
BATCH_SIZE = 64
TRAIN_START = 1000 # Number of steps to fill replay buffer before training
MEMORY_SIZE = 10000

# Training settings
EPISODES = 1000
MAX_STEPS_PER_EPISODE = 24 * 60 * 60 // 5 # Corresponds to one full day, with step interval of 5 seconds
MODEL_SAVE_PATH = 'model/dqn/elevator_dqn.h5' # Example save path for Keras/TF model

# Note: The original `setting.py` implicitly defines the number of people and their default floors.
# The `Person` class uses `np.random.randint(min_height, max_height + 1)`
# If we want fixed floors like in the original `setting.py` ([Person(4,4), Person(5,5)]),
# then min_height and max_height should be the same when creating Person instances.
# The code above reflects this for floors 4 and 5.
# Ensure `HEIGHT_OF_BUILDING` is at least 5.
# The missions are defined for floors 1, 2, 3. This implies the building must have at least 3 floors.
# Given people work on 4 and 5, it must have at least 5 floors.
# Mission target floors: 1 (off_work), 2 (breakfast, lunch), 3 (conferences).
# Person default floors: 4, 5.
# So, building height must be >= 5.
print(f"Setting.py: {len(people)} people initialized.")
print(f"Setting.py: Building height set to {HEIGHT_OF_BUILDING}, lifts to {NUM_LIFTS}.")
for i, p in enumerate(people):
    if i < 5: # Print details for a few people to check
        print(f"  Person {i}: default_layer={p.default_layer}, num_missions={len(p.missions)}")
