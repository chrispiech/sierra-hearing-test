import math
import numpy as np
import random
from constants import LING_SOUNDS

class DefaultAgent:

  def __init__(self):
    self.index = 0

  def get_next_item(self):
    sound = random.choice(LING_SOUNDS)
    difficulty = random.random()

    return {
      "difficulty": difficulty,
      "ling_sound": sound
    }
  
  def update_belief(self, item, response):
    pass
  
  def inference(self):
    prediction = {}
    for sound in LING_SOUNDS:
      prediction[sound] = 0.5

    return prediction 