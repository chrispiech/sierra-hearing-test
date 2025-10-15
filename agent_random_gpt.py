import math
import numpy as np
import random
from constants import LING_SOUNDS

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
  
    def inference(self):
        """
        This is chat gpts solution to inference.
        It uses knowledge of the simulator!
        """

        prediction = {}
        slope_db = 5.0                  # must match simulate_response slope
        prior_mean = 40.0               # shrink toward 40 when data are weak
        prior_sd = 15.0                 # amount of shrinkage
        prior_var_inv = 1.0 / (prior_sd ** 2)
        eps = 1e-9

        # Ability search grid (in dB HL)
        grid = np.arange(0.0, 100.0 + 0.5, 0.5)

        for sound in LING_SOUNDS:
            vols, ys = [], []
            for item, resp in self.observations:
                if item["ling_sound"] == sound:
                    vols.append(item["volume"])
                    ys.append(int(resp))

            if not vols:
                # No observations for this sound: fall back to prior mean
                prediction[sound] = float(prior_mean)
                continue

            # Compute difficulties using YOUR get_difficulty()
            diffs = np.array([prior_mean])  # placeholder to ensure get_difficulty is in scope
            # Recompute properly:
            Ds = np.array([get_difficulty(v) for v in vols])  # shape (N,)

            # For each candidate ability A, compute Bernoulli log-likelihood + log-prior
            best_ll = -np.inf
            best_A = prior_mean
            for A in grid:
                logits = (A - Ds) / slope_db
                p = 1.0 / (1.0 + np.exp(-logits))
                ll = (np.array(ys) * np.log(p + eps) + (1 - np.array(ys)) * np.log(1 - p + eps)).sum()
                # Gaussian log-prior on A
                lp = -0.5 * ((A - prior_mean) ** 2) * prior_var_inv
                total = ll + lp
                if total > best_ll:
                    best_ll = total
                    best_A = A

            prediction[sound] = float(best_A)

        return prediction
    
def get_difficulty(volume):
    """
    Convert stimulus volume (dB HL) to an effective 'difficulty' level (dB HL)
    used for comparison with listener ability.

    Higher volume  -> lower difficulty
    Lower volume   -> higher difficulty
    """
    # Parameters tuned to roughly match psychometric growth for speech:
    # 80% identification ≈ ability level; +10 dB → near-ceiling performance.
    midpoint = 50       # around conversational speech level
    slope = 0.2         # how fast difficulty decreases with volume

    # Map volume → normalized difficulty factor in [0,1]
    rel = 1 - sigmoid((volume - midpoint) * slope)

    # Scale to plausible dB HL difficulty (higher → harder)
    max_difficulty = 100
    min_difficulty = 0
    difficulty = min_difficulty + rel * (max_difficulty - min_difficulty)
    return difficulty

def sigmoid(x):
    return 1 / (1 + np.exp(-x))