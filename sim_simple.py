from agent_random import DefaultAgent
from tqdm import tqdm
from scipy import stats
import numpy as np
from constants import LING_SOUNDS

N_TESTS = 50000
N_ITEMS = 40

def main():
  losses = []
  for i in tqdm(range(N_TESTS)):
    agent = DefaultAgent()
    abilities = sample_abilities()
    run_adaptive_test(abilities, agent)
    guess = agent.inference()
    loss = score_guess(guess, abilities)
    losses.append(loss)
  mean = np.mean(losses)
  std_err = np.std(losses) / np.sqrt(len(losses))
  print(f'{mean} +/- {std_err}')

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


def sample_abilities():
  abilities = {}
  seed_ability = stats.norm.rvs(loc=0, scale=2)
  for ling in LING_SOUNDS:
    new_ability = seed_ability + stats.norm.rvs(loc=0, scale=1)
    abilities[ling] = new_ability
    seed_ability = new_ability
  return abilities

def simulate_response(abilities, item):
  difficulty = item["difficulty"]
  ling_index = item["ling_sound"]
  ability = abilities[ling_index]
  diff = ability - difficulty
  prob_correct = sigmoid(diff)
  return stats.bernoulli.rvs(prob_correct)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

if __name__ == '__main__':
  main()