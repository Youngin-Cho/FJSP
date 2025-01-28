import random
import numpy as np


class Heuristic:
    def __init__(self, num_jobs, num_machines):
        self.num_jobs = num_jobs
        self.num_machines = num_machines

    def act(self, state):
        priority_index = state.priority_index.flatten()
        candidates = np.where(priority_index == np.max(priority_index))[0]
        action = np.random.choice(candidates)
        return action