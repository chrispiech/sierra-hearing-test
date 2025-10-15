
from tqdm import tqdm
from scipy import stats
import numpy as np
from constants import LING_SOUNDS

def sample_abilities():
    """
    Sample prior 80%-identification levels (in dB HL) for the six Ling sounds
    for a listener with hearing loss (no inputs required).

    Returns
    -------
    dict
        {"ah": dB_HL, "oo": ..., "sh": ..., "ss": ..., "ee": ..., "mm": ...}
        representing sampled 80% identification levels (in quiet, unaided).
    """

    suprathreshold_penalty_per_dB = 0.10
    rng = np.random.default_rng()

    # Built-in common sloping HF-loss template (mild→moderate, e.g., presbycusis)
    audiogram = {250: 15, 500: 20, 1000: 25, 2000: 40, 4000: 55, 6000: 65}

    # Ensure keys are numeric and sorted
    ag_freqs = np.array(sorted(float(f) for f in audiogram.keys()))
    ag_thrs  = np.array([audiogram[float(f)] for f in ag_freqs])

    def interp_threshold(freq_hz: float) -> float:
        """Interpolate audiogram on a log-frequency axis."""
        if freq_hz <= ag_freqs[0]:
            return float(ag_thrs[0])
        if freq_hz >= ag_freqs[-1]:
            return float(ag_thrs[-1])
        x = np.log10(ag_freqs)
        y = ag_thrs
        xq = np.log10(freq_hz)
        return float(np.interp(xq, x, y))

    # ---- Ling sound → frequency weights (sum to 1.0) ----
    grid = np.array([250, 500, 1000, 2000, 4000, 6000], dtype=float)

    def w(*pairs):
        """Build a normalized weight vector on 'grid' from (freq, weight) pairs."""
        vec = np.zeros_like(grid, dtype=float)
        for f, wt in pairs:
            diffs = np.abs(grid - f)
            nearest = np.argmin(diffs)
            vec[nearest] += wt * 0.8
            if nearest - 1 >= 0:
                vec[nearest - 1] += wt * 0.1
            if nearest + 1 < len(grid):
                vec[nearest + 1] += wt * 0.1
        vec = vec / vec.sum()
        return vec

    # Spectral emphasis per Ling sound
    weights = {
        "mm": w((250, 0.7), (1000, 0.2), (2500, 0.1)),
        "oo": w((300, 0.7), (700, 0.3)),
        "ah": w((700, 0.6), (1100, 0.4)),
        "ee": w((300, 0.3), (2500, 0.7)),
        "sh": w((1800, 0.5), (5000, 0.5)),
        "ss": w((5500, 1.0)),
    }

    # Baseline (normal-hearing) 80%-ID anchor in dB HL for each sound in quiet.
    baseline_mu = {
        "ah": 5.0, "oo": 5.0, "sh": 8.5, "ss": 12.0, "ee": 5.0, "mm": 4.5
    }

    # Effective loss per sound
    thr_at_grid = np.array([interp_threshold(f) for f in grid])
    eff_loss = {s: float((wv * thr_at_grid).sum()) for s, wv in weights.items()}

    # Suprathreshold penalty (scaled to HF loss >= 2 kHz)
    hf_bins = grid >= 2000
    hf_loss_scalar = float(np.maximum(0.0, thr_at_grid[hf_bins]).mean())
    supra_penalty = suprathreshold_penalty_per_dB * (hf_loss_scalar / 10.0)  # per 10 dB HF loss

    # Mean vector (per listener) before sampling
    mu = np.array([
        baseline_mu["ah"] + eff_loss["ah"] + supra_penalty,
        baseline_mu["oo"] + eff_loss["oo"] + supra_penalty,
        baseline_mu["sh"] + eff_loss["sh"] + supra_penalty,
        baseline_mu["ss"] + eff_loss["ss"] + supra_penalty,
        baseline_mu["ee"] + eff_loss["ee"] + supra_penalty,
        baseline_mu["mm"] + eff_loss["mm"] + supra_penalty,
    ])

    # Prior SDs (dB) and correlations
    sd = np.array([3.0, 3.0, 4.0, 4.5, 3.0, 3.0])
    R = np.array([
        [1.00, 0.60, 0.30, 0.25, 0.60, 0.60],  # ah
        [0.60, 1.00, 0.30, 0.25, 0.60, 0.60],  # oo
        [0.30, 0.30, 1.00, 0.55, 0.30, 0.30],  # sh
        [0.25, 0.25, 0.55, 1.00, 0.25, 0.25],  # ss (/s/)
        [0.60, 0.60, 0.30, 0.25, 1.00, 0.60],  # ee
        [0.60, 0.60, 0.30, 0.25, 0.60, 1.00],  # mm
    ])
    S = np.outer(sd, sd) * R

    # Sample once from MVN
    sample = rng.multivariate_normal(mu, S)

    sounds = ["ah", "oo", "sh", "ss", "ee", "mm"]
    return dict(zip(sounds, sample))

def simulate_response(abilities, item):
    difficulty = get_difficulty(item["volume"])
    ling_sound = item["ling_sound"]
    ability = abilities[ling_sound]
    diff = ability - difficulty
    prob_correct = sigmoid(diff)
    return stats.bernoulli.rvs(prob_correct)

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

if __name__ == '__main__':
  main()