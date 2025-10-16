
from agent_ppo import PPOAgent
from agent_random_grid import GridAgent
from agent_random_baseline import DefaultAgent
from tqdm import tqdm
from scipy import stats
import numpy as np
from constants import LING_SOUNDS
from sim_correlation import simulate_response
from sim_correlation import sample_abilities

N_TESTS = 10000
N_ITEMS = 40
UPDATE_WEIGHTS = False 

agents = {
    'random-baseline':DefaultAgent,
    'random-grid':GridAgent,
    'ppo':PPOAgent
}

def main():
    results = {name: [] for name in agents}
    # one persistent instance per agent (they handle their own load/save)
    instances = {name: AgentClass() for name, AgentClass in agents.items()}

    for _ in tqdm(range(N_TESTS), desc="Overall progress"):
        abilities = sample_abilities()
        for name, agent in instances.items():
            # start a fresh patient/session
            agent.reset_observations()
            run_adaptive_test(abilities, agent)
            guess = agent.inference()
            results[name].append(score_guess(guess, abilities))
            if UPDATE_WEIGHTS:
                agent.observe_truth(abilities)

    # summarize
    for name, losses in results.items():
        mean = np.mean(losses)
        std_err = np.std(losses, ddof=1) / np.sqrt(len(losses))
        print(f"{name}: mean loss = {mean:.3f} ± {std_err:.3f}")

    # let each agent persist itself
    for agent in instances.values():
        if hasattr(agent, "save_policy"):
            agent.save_policy()

def score_guess(guess, abilities):
    loss = 0
    for ling in LING_SOUNDS:
        guess_i = guess[ling]
        truth_i = abilities[ling]
        loss += (guess_i - truth_i) ** 2
    return loss / len(guess)

def run_adaptive_test(abilities, agent):
    responses = []
    for i in range(N_ITEMS):
        item = agent.get_next_item()
        response = simulate_response(abilities, item)
        responses.append({
            "item": item,
            "response": response
        })
        agent.update_belief(item, response)
    return responses

if __name__ == '__main__':
  main()