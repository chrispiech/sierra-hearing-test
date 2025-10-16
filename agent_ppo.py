# transformer_ppo_agent_psycho.py
import math
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from collections import namedtuple, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from constants import LING_SOUNDS
from inference import psycho_inference  # <-- your function

# -----------------------
# Config / helpers
# -----------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default checkpoint location
DEFAULT_CKPT_DIR = Path("checkpoints")
DEFAULT_CKPT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CKPT = str(DEFAULT_CKPT_DIR / "ppo2.pt")

V_MIN, V_MAX = 20, 80
DB_STEP = 1          # tip: try 2 for the first few thousand episodes as a curriculum
VOL_BINS = (V_MAX - V_MIN) // DB_STEP + 1

SOUND_TO_ID = {s: i for i, s in enumerate(LING_SOUNDS)}
ID_TO_SOUND = {i: s for s, i in SOUND_TO_ID.items()}
N_SOUNDS = len(LING_SOUNDS)

def clamp_db(x): return float(np.clip(x, V_MIN, V_MAX))
def vol_to_bin(v_db): return int(round((clamp_db(v_db) - V_MIN) / DB_STEP))
def bin_to_vol(b): return float(V_MIN + b * DB_STEP)

def rmse(pred: dict, truth: dict):
    if not truth: return 0.0
    diffs = []
    for s, tval in truth.items():
        if s in pred and pred[s] is not None:
            diffs.append((float(pred[s]) - float(tval)) ** 2)
    if not diffs: return 0.0
    return float(math.sqrt(sum(diffs) / len(diffs)))

def safe_psycho_inference(observations_prefix):
    """
    Wraps psycho_inference to be robust to empty prefixes or occasional errors.
    Returns a dict sound->pred (float). Fallback to 40.0 per sound if needed.
    """
    try:
        if len(observations_prefix) == 0:
            return {s: 40.0 for s in LING_SOUNDS}
        pred = psycho_inference(observations_prefix)
        return {s: float(pred.get(s, 40.0)) for s in LING_SOUNDS}
    except Exception:
        return {s: 40.0 for s in LING_SOUNDS}

# -----------------------
# Tiny transformer policy
# -----------------------
@dataclass
class HP:
    d_model: int = 192
    n_layers: int = 3
    n_heads: int = 6
    dropout: float = 0.1

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01       # a bit higher to avoid instant collapse
    vf_coef: float = 0.5
    lr: float = 3e-4
    weight_decay: float = 1e-2
    ppo_epochs: int = 4
    batch_size_tokens: int = 512
    target_kl: float = 0.02
    max_grad_norm: float = 0.5

class TransformerBackbone(nn.Module):
    """ One token per trial: previous (sound, volume bin, outcome, trial_idx). """
    def __init__(self, d_model, n_layers, n_heads, dropout):
        super().__init__()
        self.emb_sound = nn.Embedding(N_SOUNDS, d_model)
        self.emb_vol   = nn.Embedding(VOL_BINS, d_model)
        self.emb_out   = nn.Embedding(3, d_model)     # 0=start, 1=correct, 2=incorrect
        self.emb_idx   = nn.Embedding(4096, d_model)

        self.fuse = nn.Linear(4*d_model, d_model)

        # post-norm to enable nested-tensor fast path (silences the warn)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dropout=dropout, batch_first=True, norm_first=False
        )
        self.tr = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, tokens):  # tokens: dict of [B,T]
        es = self.emb_sound(tokens["sound"])
        ev = self.emb_vol(tokens["vol"])
        eo = self.emb_out(tokens["outcome"])
        ei = self.emb_idx(tokens["trial_idx"])
        x = self.fuse(torch.cat([es, ev, eo, ei], dim=-1))
        # causal mask (block future)
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        x = self.tr(x, mask=mask)
        return self.ln(x)

class PolicyHeads(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.pi_sound = nn.Linear(d_model, N_SOUNDS)
        self.pi_vol   = nn.Linear(d_model, VOL_BINS)
        self.v        = nn.Linear(d_model, 1)
    def forward(self, h_last):
        return self.pi_sound(h_last), self.pi_vol(h_last), self.v(h_last).squeeze(-1)

class TransformerPolicy(nn.Module):
    def __init__(self, hp: HP):
        super().__init__()
        self.backbone = TransformerBackbone(hp.d_model, hp.n_layers, hp.n_heads, hp.dropout)
        self.heads = PolicyHeads(hp.d_model)

    def forward(self, tokens):
        h = self.backbone(tokens)
        h_last = h[:, -1, :]
        return self.heads(h_last)

    @torch.no_grad()
    def act(self, tokens):
        logits_s, logits_v, value = self.forward(tokens)
        ds, dv = Categorical(logits=logits_s), Categorical(logits=logits_v)
        a_s, a_v = ds.sample(), dv.sample()
        logp = ds.log_prob(a_s) + dv.log_prob(a_v)
        return a_s, a_v, logp, value

# -----------------------
# PPO trainer (+ metrics)
# -----------------------
Step = namedtuple("Step", "sound vol outcome trial_idx a_sound a_vol logp value reward done")

class PPOTrainer:
    def __init__(self, policy: TransformerPolicy, hp: HP):
        self.policy = policy.to(DEVICE)
        self.hp = hp
        self.opt = torch.optim.AdamW(self.policy.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

    def _iter_minibatches(self, batch):
        N = batch["sound"].shape[0]
        idx = torch.randperm(N, device=DEVICE)
        for i in range(0, N, self.hp.batch_size_tokens):
            j = idx[i:i+self.hp.batch_size_tokens]
            yield {k: v[j] for k, v in batch.items()}

    def update(self, batch):
        # returns a tiny metrics dict
        all_entropy, all_kl, all_pi, all_v = [], [], [], []
        for _ in range(self.hp.ppo_epochs):
            kl_avg, n = 0.0, 0
            for mb in self._iter_minibatches(batch):
                tokens = {
                    "sound":     mb["sound"].unsqueeze(1),
                    "vol":       mb["vol"].unsqueeze(1),
                    "outcome":   mb["outcome"].unsqueeze(1),
                    "trial_idx": mb["trial_idx"].unsqueeze(1),
                }
                logits_s, logits_v, value = self.policy(tokens)
                ds, dv = Categorical(logits=logits_s), Categorical(logits=logits_v)
                logp = ds.log_prob(mb["a_sound"]) + dv.log_prob(mb["a_vol"])
                entropy = (ds.entropy().mean() + dv.entropy().mean())

                ratio = torch.exp(logp - mb["logp"])
                adv = mb["adv"]
                loss_pi = -torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1-self.hp.clip_eps, 1+self.hp.clip_eps) * adv
                ).mean()
                loss_v  = F.huber_loss(value, mb["ret"])
                loss = loss_pi + self.hp.vf_coef * loss_v - self.hp.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.hp.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    kl = (mb["logp"] - logp).mean().clamp_min(0).item()
                    kl_avg += kl; n += 1
                    all_entropy.append(float(entropy.item()))
                    all_pi.append(float(loss_pi.item()))
                    all_v.append(float(loss_v.item()))
            if n and (kl_avg/n) > self.hp.target_kl:
                break
            if n: all_kl.append(kl_avg/n)
        def _mean(xs): return float(np.mean(xs)) if xs else 0.0
        return {
            "entropy": _mean(all_entropy),
            "approx_kl": _mean(all_kl),
            "loss_pi": _mean(all_pi),
            "loss_v": _mean(all_v),
        }

# -----------------------
# The Agent (your API)
# -----------------------
class PPOAgent:
    def __init__(self, checkpoint_path: str = DEFAULT_CKPT, autoload: bool = True):
        self.hp = HP()
        self.model = TransformerPolicy(self.hp).to(DEVICE)
        self.trainer = PPOTrainer(self.model, self.hp)

        self.observations = []   # [(item, response_int)]
        self.episode_steps = []  # PPO Step records for this exam
        self.t = 0

        # previous-token fields (start token)
        self.prev_sound_id = 0
        self.prev_vol_bin  = vol_to_bin(40.0)
        self.prev_outcome  = 0   # 0=start, 1=correct, 2=incorrect

        # cache from last action
        self._last = None

        # where to save/load by default
        self.checkpoint_path = checkpoint_path
        ckpt_dir = Path(os.path.dirname(self.checkpoint_path) or ".")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # simple rolling metrics
        self._episodes = 0
        self._ema_return = None
        self._ema_entropy = None
        self._ema_kl = None
        self._ema_improve = None
        self._ema_cover = None
        self._ema_volcost = None

        # auto-load if present
        if autoload and Path(self.checkpoint_path).exists():
            try:
                self.load_policy(self.checkpoint_path, load_optimizer=True)
                print(f"[PPOAgent] Loaded policy from {self.checkpoint_path}")
            except Exception as e:
                print(f"[PPOAgent] Warning: failed to load {self.checkpoint_path}: {e}")

    # ------------------------------------
    # Per-patient/session reset
    # ------------------------------------
    def reset_observations(self):
        """Call at the start of a new patient’s test."""
        self.observations.clear()
        self.episode_steps.clear()
        self.t = 0
        self.prev_sound_id = 0
        self.prev_vol_bin  = vol_to_bin(40.0)
        self.prev_outcome  = 0
        self._last = None

    # ------------------------------------
    # Save / Load policy
    # ------------------------------------
    def save_policy(self, path: str | None = None):
        """
        Saves model + optimizer + hyperparams for future training continuation.
        """
        path = path or self.checkpoint_path
        tmp = f"{path}.tmp"
        ckpt = {
            "version": 1,
            "torch_version": torch.__version__,
            "state_dict": self.model.state_dict(),
            "optimizer": self.trainer.opt.state_dict(),
            "hp": asdict(self.hp),
            "meta": {"n_sounds": N_SOUNDS, "vol_bins": VOL_BINS, "v_range": (V_MIN, V_MAX)},
            "trainer_meta": {
                "episodes": self._episodes,
                "ema_return": self._ema_return,
                "ema_entropy": self._ema_entropy,
                "ema_kl": self._ema_kl,
                "ema_improve": self._ema_improve,
                "ema_cover": self._ema_cover,
                "ema_volcost": self._ema_volcost,
            }
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(ckpt, tmp)
        os.replace(tmp, path)  # atomic move

    def load_policy(self, path: str | None = None, strict: bool = True, load_optimizer: bool = True):
        """
        Loads a previously saved policy checkpoint.
        """
        path = path or self.checkpoint_path
        ckpt = torch.load(path, map_location=DEVICE)

        if "hp" in ckpt and ckpt["hp"] != asdict(self.hp):
            self.hp = HP(**ckpt["hp"])
            self.model = TransformerPolicy(self.hp).to(DEVICE)
            self.trainer = PPOTrainer(self.model, self.hp)

        self.model.load_state_dict(ckpt["state_dict"], strict=strict)

        if load_optimizer and "optimizer" in ckpt:
            try:
                self.trainer.opt.load_state_dict(ckpt["optimizer"])
                for state in self.trainer.opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(DEVICE)
            except Exception:
                pass

        tm = ckpt.get("trainer_meta", {})
        self._episodes = int(tm.get("episodes", 0))
        self._ema_return = tm.get("ema_return", None)
        self._ema_entropy = tm.get("ema_entropy", None)
        self._ema_kl = tm.get("ema_kl", None)
        self._ema_improve = tm.get("ema_improve", None)
        self._ema_cover = tm.get("ema_cover", None)
        self._ema_volcost = tm.get("ema_volcost", None)

    # ----- acting -----
    def _single_token(self):
        return {
            "sound":     torch.tensor([[self.prev_sound_id]], device=DEVICE).long(),
            "vol":       torch.tensor([[self.prev_vol_bin]], device=DEVICE).long(),
            "outcome":   torch.tensor([[self.prev_outcome]], device=DEVICE).long(),
            "trial_idx": torch.tensor([[min(self.t, 4095)]], device=DEVICE).long(),
        }

    def get_next_item(self):
        with torch.no_grad():
            a_s, a_v, logp, value = self.model.act(self._single_token())
        sound_id = int(a_s.item())
        vol_bin  = int(a_v.item())
        self._last = dict(
            sound_id=sound_id, vol_bin=vol_bin,
            logp=float(logp.item()), value=float(value.item()),
            tok=dict(sound=self.prev_sound_id, vol=self.prev_vol_bin,
                     outcome=self.prev_outcome, trial_idx=min(self.t, 4095))
        )
        return {"volume": bin_to_vol(vol_bin), "ling_sound": ID_TO_SOUND[sound_id]}

    # ----- logging during session -----
    def update_belief(self, item, response):
        """
        response: 1 for correct, 0 for incorrect (int/bool accepted).
        """
        resp_int = int(response)
        self.observations.append((item, resp_int))

        st = self._last["tok"]
        self.episode_steps.append(Step(
            sound=st["sound"], vol=st["vol"], outcome=st["outcome"], trial_idx=st["trial_idx"],
            a_sound=self._last["sound_id"], a_vol=self._last["vol_bin"],
            logp=self._last["logp"], value=self._last["value"],
            reward=0.0, done=0
        ))

        # Update prev token fields for next step
        self.prev_sound_id = SOUND_TO_ID[item["ling_sound"]]
        self.prev_vol_bin  = vol_to_bin(item["volume"])
        self.prev_outcome  = 1 if resp_int == 1 else 2
        self.t += 1

    # ----- training after the exam -----
    def observe_truth(self, abilities):
        """
        Called once at end of exam.
        Reward = belief improvement (scaled) + coverage bonus - loudness penalty.
        This aligns with 'psycho_inference' quality rather than population-average proximity.
        """
        if not self.episode_steps:
            return

        # 1) Build per-step rewards
        # belief improvement (scaled up), coverage with diminishing returns, and loudness penalty.
        IMPROVE_SCALE = 10.0   # scale RMSE improvement so PPO "feels" it
        COV_WEIGHT    = 0.3    # weight for coverage bonus
        ALPHA_VOL     = 0.01   # loudness penalty

        rewards = []
        sound_counts = defaultdict(int)
        prev_err = None
        improve_terms, cover_terms, volcost_terms = [], [], []

        for k in range(len(self.observations)):
            prefix = self.observations[:k+1]
            pred_k = safe_psycho_inference(prefix)
            err_k = rmse(pred_k, abilities)
            improve = 0.0 if prev_err is None else (prev_err - err_k)
            improve_scaled = IMPROVE_SCALE * improve

            s = prefix[-1][0]["ling_sound"]
            sound_counts[s] += 1
            # diminishing returns: bonus when a sound is under-sampled
            cover = 1.0 / math.sqrt(sound_counts[s])

            v = clamp_db(prefix[-1][0]["volume"])
            vol_cost = ALPHA_VOL * (v - V_MIN) / (V_MAX - V_MIN)

            r = float(improve_scaled + COV_WEIGHT * cover - vol_cost)
            rewards.append(r)

            improve_terms.append(float(improve_scaled))
            cover_terms.append(float(COV_WEIGHT * cover))
            volcost_terms.append(float(vol_cost))
            prev_err = err_k

        # 2) Mark terminal
        steps = list(self.episode_steps)
        steps[-1] = steps[-1]._replace(done=1)
        steps = [steps[i]._replace(reward=rewards[i]) for i in range(len(steps))]

        # 3) Compute GAE / returns -> flat batch
        gamma, lam = self.hp.gamma, self.hp.gae_lambda
        vals = [s.value for s in steps] + [0.0]
        dones = [s.done for s in steps] + [1]
        adv, last = [0.0]*len(steps), 0.0
        ep_return = 0.0
        for t in reversed(range(len(steps))):
            nonterm = 1 - dones[t+1]
            delta = steps[t].reward + gamma * vals[t+1] * nonterm - vals[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
            ep_return += steps[t].reward
        ret = [adv[t] + vals[t] for t in range(len(steps))]

        def tens(field, dtype=torch.long):
            arr = torch.tensor([getattr(s, field) for s in steps], device=DEVICE)
            return arr.to(dtype)

        batch = {
            "sound":     tens("sound"),
            "vol":       tens("vol"),
            "outcome":   tens("outcome"),
            "trial_idx": tens("trial_idx"),
            "a_sound":   tens("a_sound"),
            "a_vol":     tens("a_vol"),
            "logp":      tens("logp", torch.float),
            "value":     tens("value", torch.float),
            "adv":       torch.tensor(adv, device=DEVICE, dtype=torch.float),
            "ret":       torch.tensor(ret, device=DEVICE, dtype=torch.float),
        }
        batch["adv"] = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std(unbiased=False) + 1e-8)

        # 4) PPO update + metrics
        metrics = self.trainer.update(batch)

        # EMAs for telemetry
        self._episodes += 1
        def ema(prev, x, k=0.05):  # 5% smoothing
            return x if prev is None else (1-k)*prev + k*x
        self._ema_return  = ema(self._ema_return, ep_return)
        self._ema_entropy = ema(self._ema_entropy, metrics["entropy"])
        self._ema_kl      = ema(self._ema_kl, metrics["approx_kl"])
        self._ema_improve = ema(self._ema_improve, float(np.mean(improve_terms)))
        self._ema_cover   = ema(self._ema_cover, float(np.mean(cover_terms)))
        self._ema_volcost = ema(self._ema_volcost, float(np.mean(volcost_terms)))

        # Very slow entropy anneal (keep some exploration)
        self.hp.ent_coef = max(0.005, self.hp.ent_coef * 0.999)

        if (self._episodes % 50) == 0:
            print(f"[PPO] ep={self._episodes:5d} "
                  f"ret_ema={self._ema_return:6.3f} "
                  f"ent_ema={self._ema_entropy:5.3f} "
                  f"kl_ema={self._ema_kl:5.3f} "
                  f"impr_ema={self._ema_improve:6.3f} "
                  f"cov_ema={self._ema_cover:5.3f} "
                  f"vol_ema={self._ema_volcost:5.3f}")

        # 5) Reset for next exam
        self.reset_observations()

    def inference(self):
        # Always use your psychometric inference for a current estimate
        return safe_psycho_inference(self.observations)