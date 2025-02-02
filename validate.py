import os
import torch
import pandas as pd

from environment.fjsp import FlexibleJobShop


def evaluate(agent, device, config):
    agent.policy.eval()
    val_paths = os.listdir(config.val_dir)

    with torch.no_grad():
        total_tardiness_lst = []
        for path in val_paths:
            env = FlexibleJobShop(config.val_dir + path, config.look_ahead, device, record_events=config.record_events)

            state = env.reset()

            while True:
                action, log_prob, state_value = agent.get_action(state)
                next_state, reward, done = env.step(action)

                state = next_state

                if done:
                    break

            total_tardiness_lst.append(env.model['Sink'].total_tardiness)

        total_tardiness_avg = sum(total_tardiness_lst) / len(total_tardiness_lst)

        return total_tardiness_avg