import numpy as np

from constants import LING_SOUNDS
from sim_correlation import get_difficulty

def psycho_inference(observations):
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
        for item, resp in observations:
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