import math
import numpy as np
import random
from constants import LING_SOUNDS

class DefaultAgent:

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
  
    def inference(self):
        prediction = {}
        for sound in LING_SOUNDS:
            prediction[sound] = 40
        return prediction 