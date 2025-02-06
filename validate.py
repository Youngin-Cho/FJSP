import os
import torch
import pandas as pd

from environment.fjsp import FlexibleJobShop


def evaluate(agent, device, config):
    agent.policy.eval()
    val_paths = os.listdir(config.val_dir)

    with torch.no_grad():
        makespan_lst = []
        # total_tardiness_lst = []
        for path in val_paths:
            env = FlexibleJobShop(config.val_dir + path, config.look_ahead, device,
                                  state_encoding=config.state_encoding, record_events=config.record_events)

            state = env.reset()

            while True:
                action, log_prob, state_value = agent.get_action(state)
                next_state, reward, done = env.step(action)

                state = next_state

                if done:
                    break

            makespan_lst.append(env.model['Sink'].completion_time)
            # total_tardiness_lst.append(env.model['Sink'].total_tardiness)

        makespan_avg = sum(makespan_lst) / len(makespan_lst)
        # total_tardiness_avg = sum(total_tardiness_lst) / len(total_tardiness_lst)

        return makespan_avg
        # return total_tardiness_avg