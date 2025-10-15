import math
import numpy as np
import random
from constants import LING_SOUNDS

from inference import psycho_inference
class GridAgent:

    def __init__(self):
        self.observations = []

    def get_next_item(self):
        sound = random.choice(LING_SOUNDS)
        volume = np.random.uniform(20, 80)

        return {
            "volume": volume,
            "ling_sound": sound
        }
  
    def update_belief(self, item, response):
        self.observations.append((item, response))

    def reset_observations(self):
        self.observations = []

    def observe_truth(self, abilities):
        # this is your chance to update your policy
        pass
  
    def inference(self):
        return psycho_inference(self.observations)
    