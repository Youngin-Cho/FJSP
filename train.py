import os
import json
import argparse
import torch
import numpy as np

from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from environment.data import DataGenerator
from environment.fjsp import FlexibleJobShop, State
from agent.ppo import Agent
from validate import evaluate


def get_config():
    parser = argparse.ArgumentParser(description="FJSP")

    parser.add_argument("--no_vessl", action='store_true', help="Disable VESSL")
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    parser.add_argument("--load_model", type=int, default=0, help="whether to load the trained model (0: False, 1:True)")
    parser.add_argument("--model_path", type=str, default=None, help="model file path")

    parser.add_argument("--n_jobs", type=int, default=10, help="number of jobs")
    parser.add_argument("--n_init_jobs", type=int, default=5, help="number of jobs")
    parser.add_argument("--n_machines", type=int, default=5, help="number of machines")
    parser.add_argument("--n_operations_min", type=int, default=4, help="minimum number of operations per job")
    parser.add_argument("--n_operations_max", type=int, default=6, help="maximum number of operations per job")
    parser.add_argument("--n_options_min", type=int, default=1, help="minimum number of available machines")
    parser.add_argument("--n_options_max", type=int, default=5, help="maximum number of available machines")
    parser.add_argument("--proctime_min", type=int, default=1, help="minimum processing time")
    parser.add_argument("--proctime_max", type=int, default=20, help="maximum processing time")
    parser.add_argument("--iat_avg", type=float, default=4, help="average inter-arrival time")
    parser.add_argument("--ddt", type=float, default=1.2, help="due date tardiness")

    parser.add_argument("--look_ahead", type=int, default=2, help="look-ahead parameter")
    parser.add_argument("--embed_dim", type=int, default=128, help="node embedding dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="number of heads in MHA sub-layers")
    parser.add_argument("--n_layers_ff", type=int, default=2, help="number of FFN layers")
    parser.add_argument("--n_layers_hgt", type=int, default=2, help="number of MLAN layers")
    parser.add_argument("--n_layers_actor", type=int, default=2, help="number of Actor layers")
    parser.add_argument("--n_layers_critic", type=int, default=2, help="number of Critic layers")
    parser.add_argument('--hidden_dim_actor', type=int, default=384, help='Dimension of hidden layers in Actor')
    parser.add_argument('--hidden_dim_critic', type=int, default=256, help='Dimension of hidden layers in Critic')

    parser.add_argument("--n_episodes", type=int, default=1000, help="number of episodes")
    parser.add_argument("--n_envs_RL", type=int, default=20, help="number of environments")
    parser.add_argument("--n_envs_SPT", type=int, default=0, help="number of environments")
    parser.add_argument("--n_envs_MDD", type=int, default=0, help="number of environments")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--lr_decay", type=float, default=1.0, help="learning rate decay ratio")
    parser.add_argument("--lr_step", type=int, default=250, help="step size to reduce learning rate")
    parser.add_argument('--max_grad_norm', type=float, default=0.0,
                        help='Maximum L2 norm for gradient clipping, default 1.0 (0 to disable clipping)')
    parser.add_argument("--gamma", type=float, default=1.00, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.2, help="clipping parameter")
    parser.add_argument("--K_epoch", type=int, default=3, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=10, help="the number of steps to obtain samples")
    parser.add_argument("--P_coeff", type=float, default=1, help="coefficient for policy loss")
    parser.add_argument("--V_coeff", type=float, default=0.5, help="coefficient for value loss")
    parser.add_argument("--E_coeff", type=float, default=0.01, help="coefficient for entropy loss")

    parser.add_argument("--eval_every", type=int, default=50, help="Evaluate every x episodes")
    parser.add_argument("--save_every", type=int, default=100, help="Save a model every x episodes")
    parser.add_argument("--reset_every", type=int, default=1, help="Generate new instances every x episodes")
    parser.add_argument("--record_events", type=int, default=0, help="whether to record the events (0: False, 1:True)")

    parser.add_argument("--val_dir", type=str, default=None, help="directory where the validation data are stored")

    return parser.parse_args()


if __name__ == "__main__":
    date = datetime.now().strftime('%m%d_%H_%M')
    config = get_config()

    use_vessl = not config.no_vessl
    use_cuda = torch.cuda.is_available() and not config.no_cuda
    device = torch.device("cuda" if use_cuda else "cpu")

    load_model = bool(config.load_model)
    model_path = config.model_path

    n_episodes = config.n_episodes
    n_envs_RL = config.n_envs_RL
    n_envs_SPT = config.n_envs_SPT
    n_envs_MDD = config.n_envs_MDD
    n_envs = n_envs_RL + n_envs_SPT + n_envs_MDD

    eval_every = config.eval_every
    save_every = config.save_every
    reset_every = config.reset_every
    record_events = bool(config.record_events)

    val_dir = config.val_dir

    if use_vessl:
        import vessl
        vessl.init(organization="snu-eng-dgx", project="scheduling", hp=config)

    if use_vessl:
        model_dir = '/output/train/' + date + '/model/'
    else:
        model_dir = './output/train/' + date + '/model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if use_vessl:
        log_dir = '/output/train/' + date + '/log/'
    else:
        log_dir = './output/train/' + date + '/log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if record_events:
        if use_vessl:
            simulation_dir = '/output/train/simulation/'
        else:
            simulation_dir = '../output/train/simulation/'
        if not os.path.exists(simulation_dir):
           os.makedirs(simulation_dir)

    with open(log_dir + "parameters.json", 'w') as f:
        json.dump(vars(config), f, indent=4)

    data_generator = DataGenerator(config)
    # data_instance = data_generator.generate()

    envs_RL = [FlexibleJobShop(data_generator, config.look_ahead, device, record_events=record_events) for _ in range(n_envs_RL)]
    envs_SPT = [FlexibleJobShop(data_generator, config.look_ahead, device, record_events=record_events, guide="SPT") for _ in range(n_envs_SPT)]
    envs_MDD = [FlexibleJobShop(data_generator, config.look_ahead, device, record_events=record_events, guide="MDD") for _ in range(n_envs_MDD)]
    envs = envs_RL + envs_SPT + envs_MDD

    agent = Agent(envs[0].meta_data, envs[0].num_nodes, envs[0].input_dim_g, envs[0].input_dim_pair, config, device)

    if not use_vessl:
        writer = SummaryWriter(log_dir)

    if load_model:
        checkpoint = torch.load(model_path)
        start_episode = checkpoint['episode'] + 1
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        start_episode = 1

    with open(log_dir + "train_log.csv", 'w') as f:
        f.write('episode, reward, loss, lr\n')
    with open(log_dir + "validation_log.csv", 'w') as f:
        f.write('episode, total tardiness\n')

    for e in range(start_episode, n_episodes + 1):
        if use_vessl:
            vessl.log(payload={"Train/learnig_rate": agent.scheduler.get_last_lr()[0]}, step=e)
        else:
            writer.add_scalar("Training/Learning Rate", agent.scheduler.get_last_lr()[0], e)

        n_update = 0
        loss_episode = 0.0
        grad_norms_episode = 0.0
        grad_norms_clipped_episode = 0.0

        state_lst = [envs[i].reset() for i in range(n_envs)]
        reward_lst = [0.0 for i in range(n_envs)]
        done_lst = [False for i in range(n_envs)]
        action_flags = np.array([True for i in range(n_envs)])

        if n_envs == n_envs_RL:
            guide_flags = None
        else:
            guide_flags = np.array([False for i in range(n_envs_RL)]
                                   + [True for i in range(n_envs_SPT)]
                                   + [True for i in range(n_envs_MDD)])

        while not all(done_lst):
            update_flags = [[] for _ in range(n_envs)]
            for t in range(config.T_horizon):
                action, log_prob, state_value = agent.get_action(state_lst, action_flags, guide_flags)
                action_idx = 0
                for i, env in enumerate(envs):
                    if done_lst[i]:
                        next_state, reward, done = State(env.num_jobs, env.num_machines, config.look_ahead, device), 0.0, True
                        agent.collect_sample(i, state_lst[i], 0, 0.0, 0.0, True, 0.0)
                        update_flags[i].append(False)
                    else:
                        next_state, reward, done = env.step(action[action_idx])
                        agent.collect_sample(i, state_lst[i], action[action_idx], reward, state_value[action_idx], done, log_prob[action_idx])
                        update_flags[i].append(True)

                    if action_flags[i]:
                        action_idx += 1

                    state_lst[i] = next_state
                    reward_lst[i] += reward
                    done_lst[i] = done

                    if done:
                        action_flags[i] = False

                if all(done_lst):
                    break

            loss, grad_norms, grad_norms_clipped = agent.update(state_lst, update_flags)

            n_update += 1
            loss_episode += loss
            grad_norms_episode += grad_norms
            grad_norms_clipped_episode += grad_norms_clipped

        agent.scheduler.step()

        print("episode: %d | reward: %.4f | loss: %.4f | grad_norms: %.4f | grad_norms_clipped: %.4f"
              % (e, np.mean(reward_lst), loss_episode / n_update, grad_norms_episode / n_update, grad_norms_clipped_episode / n_update))
        with open(log_dir + "train_log.csv", 'a') as f:
            f.write('%d, %1.4f, %1.4f, %f\n' % (e, np.mean(reward_lst), loss_episode / n_update, agent.scheduler.get_last_lr()[0]))

        if use_vessl:
            vessl.log(payload={"Train/Reward": np.mean(reward_lst),
                               "Train/Loss": loss_episode / n_update,
                               "Train/GradNorms": grad_norms_episode / n_update,
                               "Train/GradNormsClipped": grad_norms_clipped_episode / n_update}, step=e)
        else:
            writer.add_scalar("Training/Reward", np.mean(reward_lst), e)
            writer.add_scalar("Training/Loss", loss_episode / n_update, e)
            writer.add_scalar("Training/GradNorms", grad_norms_episode / n_update, e)
            writer.add_scalar("Training/GradNormsClipped", grad_norms_clipped_episode / n_update, e)

        if e == start_episode or e % eval_every == 0:
            total_tardiness_avg = evaluate(agent, device, config)

            with open(log_dir + "validation_log.csv", 'a') as f:
                f.write('%d, %1.4f\n' % (e, total_tardiness_avg))

            if use_vessl:
                vessl.log(payload={"Perf/TotalTardiness": total_tardiness_avg}, step=e)
            else:
                writer.add_scalar("Validation/TotalTardiness", total_tardiness_avg, e)

        if e % save_every == 0:
            agent.save(e, model_dir)

        if e % reset_every == 0:
            # data_instance = data_generator.generate()
            envs = [FlexibleJobShop(data_generator, config.look_ahead, device, record_events=record_events) for _ in range(n_envs)]

    if not use_vessl:
        writer.close()