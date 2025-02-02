import random
import numpy as np


class Heuristic:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines

    def act(self, state):
        mask_pair = state.mask_pair.flatten()
        priority_index = state.priority_index.flatten()
        priority_index[~mask_pair] = 0.0
        candidates = np.where(priority_index == np.max(priority_index))[0]
        action = np.random.choice(candidates)
        return action