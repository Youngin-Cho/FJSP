import os
import time
import json
import random
import argparse
import torch
import pandas as pd

from torch.distributions.categorical import Categorical
from environment.fjsp import FlexibleJobShop
from agent.network import SchedulingNetwork
from agent.ppo import convert_state
from baseline.heuristics import Heuristic


def get_config():
    parser = argparse.ArgumentParser(description="FJSP")

    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument("--model_path", type=str, default=None, help="model file path")
    parser.add_argument("--param_path", type=str, default=None, help="hyper-parameter file path")
    parser.add_argument("--data_dir", type=str, default=None, help="test data path")
    parser.add_argument("--res_dir", type=str, default=None, help="test result file path")
    parser.add_argument("--sim_dir", type=str, default=None, help="simulation log file path")

    parser.add_argument("--look_ahead", type=int, default=2, help="look-ahead parameter")

    parser.add_argument("--record_events", type=int, default=0, help="whether to record the events (0: False, 1:True)")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")

    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()

    use_cuda = torch.cuda.is_available() and not config.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    model_path = config.model_path
    param_path = config.param_path
    data_dir = [config.data_dir]
    res_dir = [config.res_dir]
    sim_dir = [config.sim_dir]

    look_ahead = config.look_ahead

    record_events = bool(config.record_events)
    random_seed = config.random_seed

    for res_dir_temp in res_dir:
        if not os.path.exists(res_dir_temp):
            os.makedirs(res_dir_temp)

    algorithm = ["SPT", "MWKR", "MOR"]

    for data_dir_temp, res_dir_temp in zip(data_dir, res_dir):
        test_paths = os.listdir(data_dir_temp)
        index = ["P%d" % i for i in range(1, len(test_paths) + 1)] + ["avg"]
        columns = algorithm

        df_makespan = pd.DataFrame(index=index, columns=columns)
        # df_tardiness = pd.DataFrame(index=index, columns=columns)
        df_computing_time = pd.DataFrame(index=index, columns=columns)

        for name in columns:
            progress = 0
            list_makespan = []
            # list_tardiness = []
            list_computing_time = []

            for prob, path in zip(index, test_paths):
                random.seed(random_seed)

                data_src = data_dir_temp + path
                env = FlexibleJobShop(data_src, config.look_ahead, device,
                                      algorithm=name, record_events=config.record_events)

                if name == "RL":
                    with open(param_path, 'r') as f:
                        parameters = json.load(f)

                    config.embed_dim = parameters['embed_dim']
                    config.n_heads = parameters['n_heads']
                    config.n_layers_hgt = parameters['n_layers_hgt']
                    config.n_layers_ff = parameters['n_layers_ff']

                    config.n_layers_actor = parameters['n_layers_actor']
                    config.hidden_dim_actor = parameters['hidden_dim_actor']
                    config.n_layers_critic = parameters['n_layers_critic']
                    config.hidden_dim_critic = parameters['hidden_dim_critic']

                    agent = SchedulingNetwork(meta_data=env.meta_data,
                                              num_nodes=env.num_nodes,
                                              input_dim_g=env.input_dim_g,
                                              input_dim_pair=env.input_dim_pair,
                                              config=config).to(device)
                    checkpoint = torch.load(model_path, map_location=torch.device(device))
                    agent.load_state_dict(checkpoint['model_state_dict'])
                else:
                    agent = Heuristic(env.num_jobs, env.num_machines)

                start = time.time()
                state = env.reset()
                done = False

                while not done:
                    if name == "RL":
                        with torch.no_grad():
                            fea_g, fea_pair, mask_pair = convert_state(state, device)
                            probs, value = agent(fea_g, fea_pair, mask_pair)

                        dist = Categorical(probs)
                        action = dist.sample().item()
                    else:
                        action = agent.act(state)

                    next_state, reward, done = env.step(action)

                    state = next_state

                    if done:
                        finish = time.time()
                        makespan = env.model['Sink'].completion_time
                        # tardiness = env.model['Sink'].total_tardiness
                        computing_time = finish - start
                        break

                list_makespan.append(makespan)
                # list_tardiness.append(tardiness)
                list_computing_time.append(computing_time)

                progress += 1
                print("%d/%d test for %s done" % (progress, len(index) - 1, name))

            df_makespan[name] = list_makespan + [sum(list_makespan) / len(list_makespan)]
            # df_tardiness[name] = list_tardiness + [sum(list_tardiness) / len(list_tardiness)]
            df_computing_time[name] = list_computing_time + [sum(list_computing_time) / len(list_computing_time)]
            print("==========test for %s finished==========" % name)

        writer = pd.ExcelWriter(res_dir_temp + 'test_results.xlsx')
        df_makespan.to_excel(writer, sheet_name="makespan")
        # df_tardiness.to_excel(writer, sheet_name="tardiness")
        df_computing_time.to_excel(writer, sheet_name="computing_time")
        writer.close()