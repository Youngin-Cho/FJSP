import math
import torch
import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, OneCycleLR
from torch.distributions.categorical import Categorical
from torch.nn.functional import smooth_l1_loss
from torch_geometric.data import Batch
from agent.network import SchedulingNetwork


def convert_state(state, device, flags=None):
    if type(state) is list:
        fea_g = Batch.from_data_list([state[i].fea_g for i in range(len(state)) if flags[i]])
        fea_pair = torch.FloatTensor(np.stack([state[i].fea_pair for i in range(len(state)) if flags[i]], axis=0)).to(device)
        mask_pair = torch.from_numpy(np.stack([state[i].mask_pair for i in range(len(state)) if flags[i]], axis=0)).to(device)
        current_o = torch.LongTensor(np.stack([state[i].current_o for i in range(len(state)) if flags[i]], axis=0)).to(device)
    else:
        fea_g = Batch.from_data_list([state.fea_g])
        fea_pair = torch.FloatTensor(np.array([state.fea_pair])).to(device)
        mask_pair = torch.BoolTensor(np.array([state.mask_pair])).to(device)
        current_o = torch.LongTensor(np.array([state.current_o])).to(device)

    return fea_g, fea_pair, mask_pair, current_o


def clip_grad_norms(param_groups, max_norm=math.inf):
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


class RollOutMemory:
    def __init__(self, n_envs, device):
        self.n_envs = n_envs
        self.device = device

        # input variables
        self.fea_gs = [[] for _ in range(n_envs)]
        self.fea_pairs = [[] for _ in range(n_envs)]
        self.mask_pairs = [[] for _ in range(n_envs)]
        self.current_os = [[] for _ in range(n_envs)]

        # other variables
        self.actions = [[] for _ in range(n_envs)]
        self.rewards = [[] for _ in range(n_envs)]
        self.values = [[] for _ in range(n_envs)]
        self.dones = [[] for _ in range(n_envs)]
        self.log_probs = [[] for _ in range(n_envs)]

    def clear(self):
        # input variables
        self.fea_gs = [[] for _ in range(self.n_envs)]
        self.fea_pairs = [[] for _ in range(self.n_envs)]
        self.mask_pairs = [[] for _ in range(self.n_envs)]
        self.current_os = [[] for _ in range(self.n_envs)]

        # other variables
        self.actions = [[] for _ in range(self.n_envs)]
        self.rewards = [[] for _ in range(self.n_envs)]
        self.values = [[] for _ in range(self.n_envs)]
        self.dones = [[] for _ in range(self.n_envs)]
        self.log_probs = [[] for _ in range(self.n_envs)]

    def put(self, env_id, state, action, reward, value, done, log_probs):
        # input variables
        self.fea_gs[env_id].append(state.fea_g)
        self.fea_pairs[env_id].append(state.fea_pair)
        self.mask_pairs[env_id].append(state.mask_pair)
        self.current_os[env_id].append(state.current_o)

        # other variables
        self.actions[env_id].append(action)
        self.rewards[env_id].append(reward)
        self.values[env_id].append(value)
        self.dones[env_id].append(done)
        self.log_probs[env_id].append(log_probs)

    def get(self, flags):
        fea_gs = Batch.from_data_list([item for row in self.fea_gs for item in row])
        fea_pairs = torch.FloatTensor(np.stack(self.fea_pairs, axis=0)).to(self.device)
        mask_pairs = torch.from_numpy(np.stack(self.mask_pairs, axis=0)).to(self.device)
        current_os = torch.LongTensor(np.stack(self.current_os, axis=0)).to(self.device)

        actions = torch.LongTensor(np.stack(self.actions, axis=0)).to(self.device)
        rewards = torch.FloatTensor(np.stack(self.rewards, axis=0)).to(self.device)
        values = torch.FloatTensor(np.stack(self.values, axis=0)).to(self.device)
        dones = torch.FloatTensor(~np.stack(self.dones, axis=0)).to(self.device)
        log_probs = torch.FloatTensor(np.stack(self.log_probs, axis=0)).to(self.device)

        return fea_gs, fea_pairs, mask_pairs, current_os, actions, rewards, values, dones, log_probs


class Agent:
    def __init__(self, meta_data, num_nodes, input_dim_g, input_dim_pair, config, device):
        self.device = device

        self.n_envs = config.n_envs
        self.max_grad_norm = config.max_grad_norm
        self.lr = config.lr
        self.lr_decay = config.lr_decay
        self.lr_step = config.lr_step
        self.gamma = config.gamma
        self.lmbda = config.lmbda
        self.eps_clip = config.eps_clip
        self.K_epoch = config.K_epoch
        self.T_horizon = config.T_horizon
        self.no_adv_norm = config.no_adv_norm
        self.P_coeff = config.P_coeff
        self.V_coeff = config.V_coeff
        self.E_coeff = config.E_coeff

        self.memory = RollOutMemory(self.n_envs, device)
        self.policy = SchedulingNetwork(meta_data, num_nodes, input_dim_g, input_dim_pair, config).to(device)
        self.optimizer = Adam(self.policy.parameters(), lr=self.lr)
        self.scheduler = StepLR(optimizer=self.optimizer, step_size=self.lr_step, gamma=self.lr_decay)
        # self.scheduler = OneCycleLR(optimizer=self.optimizer, max_lr=0.00001, total_steps=1000, anneal_strategy='linear')

    def collect_sample(self, env_id, state, action, reward, value, done, log_probs):
        self.memory.put(env_id, state, action, reward, value, done, log_probs)

    def get_action(self, state, action_flags=None):
        fea_g, fea_pair, mask_pair, current_o = convert_state(state, self.device, action_flags)

        self.policy.eval()
        with torch.no_grad():
            probs, value = self.policy(fea_g, fea_pair, mask_pair, current_o)

        dist = Categorical(probs)
        action = dist.sample()
        log_probs = dist.log_prob(action)

        if type(state) is list:
            return action.cpu().numpy(), log_probs.cpu().numpy(), value.squeeze(-1).cpu().numpy()
        else:
            return action.cpu().numpy()[0], log_probs.cpu().numpy()[0], value.squeeze(-1).cpu().numpy()[0]

    def update(self, last_state, flags):
        fea_gs, fea_pairs, mask_pairs, current_os, actions, rewards, values, dones, log_probs = self.memory.get(flags)

        flags = torch.BoolTensor(flags).to(self.device)

        with torch.no_grad():
            fea_g, fea_pair, mask_pair, current_o \
                = convert_state(last_state, self.device, flags=[True for _ in range(self.n_envs)])
            _, last_value = self.policy(fea_g, fea_pair, mask_pair, current_o)
            last_value = last_value * dones[:, -1:]

        values = torch.cat((values, last_value), dim=-1)
        td_target = rewards + self.gamma * values[:, 1:] * dones
        delta = td_target - values[:, :-1]

        n_envs, len_trajectory = delta.size()
        advantages = torch.zeros((n_envs, len_trajectory)).to(self.device)
        advantage = torch.zeros(n_envs).to(self.device)

        for i in reversed(range(len_trajectory)):
            advantage = self.gamma * self.lmbda * advantage + delta[:, i]
            advantages[:, i] = advantage

        if not self.no_adv_norm:
            advantages = ((advantages - advantages.mean(dim=1, keepdim=True))
                          / (advantages.std(dim=1, correction=0, keepdim=True) + 1e-8))

        avg_loss = 0.0
        avg_policy_loss, avg_value_loss, avg_entropy_loss = 0.0, 0.0, 0.0
        avg_grad_norms, avg_grad_norms_clipped = 0.0, 0.0

        self.policy.train()
        for _ in range(self.K_epoch):
            new_probs, new_values = self.policy(fea_graph=fea_gs,
                                                fea_pair=fea_pairs.flatten(0, 1),
                                                mask_pair=mask_pairs.flatten(0, 1),
                                                current_o=current_os.flatten(0, 1))

            dist = Categorical(new_probs[flags.flatten().unsqueeze(-1).expand_as(new_probs)])
            new_log_probs = dist.log_prob(actions[flags])
            entropys = dist.entropy()

            ratio = torch.exp(new_log_probs - log_probs[flags])

            surr1 = ratio * advantages[flags]
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[flags]

            policy_loss = - torch.min(surr1, surr2)
            value_loss = smooth_l1_loss(new_values[flags.flatten()].squeeze(-1), td_target[flags].flatten())
            entropy_loss = - entropys

            loss = self.P_coeff * policy_loss + self.V_coeff * value_loss + self.E_coeff * entropy_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            grad_norms, grad_norms_clipped = clip_grad_norms(self.optimizer.param_groups, self.max_grad_norm)
            self.optimizer.step()

            avg_loss += loss.mean().item()
            avg_policy_loss += policy_loss.mean().item()
            avg_value_loss += value_loss.mean().item()
            avg_entropy_loss += entropy_loss.mean().item()
            avg_grad_norms += grad_norms[0].item()
            if isinstance(grad_norms_clipped[0], float):
                avg_grad_norms_clipped += grad_norms_clipped[0]
            else:
                avg_grad_norms_clipped += grad_norms_clipped[0].item()

        self.memory.clear()

        return (avg_loss / self.K_epoch,
                avg_policy_loss / self.K_epoch,
                avg_value_loss / self.K_epoch,
                avg_entropy_loss / self.K_epoch,
                avg_grad_norms / self.K_epoch,
                avg_grad_norms_clipped / self.K_epoch)

    def save(self, episode, model_dir):
        torch.save({"episode": episode,
                    "model_state_dict": self.policy.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict()},
                   model_dir + "episode-%d.pt" % episode)

