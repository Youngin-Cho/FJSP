import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HGTConv
from agent.layers import Transformer


class SchedulingNetwork(nn.Module):
    def __init__(self, meta_data, num_nodes, input_dim_g, input_dim_pair, config):
        super(SchedulingNetwork, self).__init__()
        self.meta_data = meta_data
        self.num_nodes = num_nodes
        self.input_dim_g = input_dim_g
        self.input_dim_pair = input_dim_pair

        self.embed_dim = config.embed_dim
        self.n_heads = config.n_heads
        self.n_layers_hgt = config.n_layers_hgt
        # self.n_layers_transformer = config.n_layers_transformer
        self.n_layers_ff = config.n_layers_ff
        # self.hidden_dim_ff = config.hidden_dim_ff

        self.n_layers_actor = config.n_layers_actor
        self.hidden_dim_actor = config.hidden_dim_actor
        self.n_layers_critic = config.n_layers_critic
        self.hidden_dim_critic = config.hidden_dim_critic

        self.conv = nn.ModuleList()
        for i in range(self.n_layers_hgt):
            if i == 0:
                self.conv.append(HGTConv(self.input_dim_g, self.embed_dim, meta_data, heads=self.n_heads))
            else:
                self.conv.append(HGTConv(self.embed_dim, self.embed_dim, meta_data, heads=self.n_heads))

        # self.embed = nn.Linear(self.input_dim_pair, self.embed_dim)
        # self.transformer = nn.ModuleList()
        # for i in range(self.n_layers_transformer):
        #     self.transformer.append(Transformer(self.n_heads, self.embed_dim, self.embed_dim, self.n_layers_ff, self.hidden_dim_ff))

        self.ffn = nn.ModuleList()
        for i in range(self.n_layers_ff):
            if i == 0:
                self.ffn.append(nn.Linear(self.input_dim_pair, self.embed_dim))
            else:
                self.ffn.append(nn.Linear(self.embed_dim, self.embed_dim))

        self.actor = nn.ModuleList()
        for i in range(self.n_layers_actor):
            if i == 0:
                self.actor.append(nn.Linear(self.embed_dim * 5, self.hidden_dim_actor))
            elif 0 < i < self.n_layers_actor - 1:
                self.actor.append(nn.Linear(self.hidden_dim_actor, self.hidden_dim_actor))
            else:
                self.actor.append(nn.Linear(self.hidden_dim_actor, 1))

        self.critic = nn.ModuleList()
        for i in range(self.n_layers_critic):
            if i == 0:
                self.critic.append(nn.Linear(self.embed_dim * 2, self.hidden_dim_critic))
            elif i < self.n_layers_critic - 1:
                self.critic.append(nn.Linear(self.hidden_dim_critic, self.hidden_dim_critic))
            else:
                self.critic.append(nn.Linear(self.hidden_dim_critic, 1))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, fea_graph, fea_pair, mask_pair, current_o):
        batch_size = fea_graph.num_graphs
        x_dict, edge_index_dict = fea_graph.x_dict, fea_graph.edge_index_dict

        for i in range(self.n_layers_hgt):
            x_dict = self.conv[i](x_dict, edge_index_dict)
            x_dict = {key: F.elu(x) for key, x in x_dict.items()}

        h_machines = x_dict["machine"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        if "operation" in self.meta_data[0]:
            h_operations = x_dict["operation"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)
        elif "job" in self.meta_data[0]:
            h_jobs = x_dict["job"].unsqueeze(0).reshape(batch_size, -1, self.embed_dim)

        h_machines_pooled = h_machines.mean(dim=-2)
        if "operation" in self.meta_data[0]:
            h_operations_pooled = h_operations.mean(dim=-2)
            jobs_gather = current_o.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            h_jobs = h_operations.gather(1, jobs_gather)
        elif "job" in self.meta_data[0]:
            h_jobs_pooled = h_jobs.mean(dim=-2)

        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, self.num_nodes["machine"], -1)
        h_machines_padding = h_machines.unsqueeze(-3).expand_as(h_jobs_padding)

        # h_pairs = self.embed(fea_pair.flatten(1, 2))
        # for i in range(self.n_layers_transformer):
        #     h_pairs = self.transformer[i](h_pairs)
        # h_pairs = h_pairs.reshape(batch_size, self.num_nodes["job"], self.num_nodes["machine"], -1)

        h_pairs = fea_pair
        for i in range(self.n_layers_ff):
            h_pairs = self.ffn[i](h_pairs)
            h_pairs = F.elu(h_pairs)

        h_machines_pooled_padding = h_machines_pooled[:, None, None, :].expand_as(h_machines_padding)

        if "operation" in self.meta_data[0]:
            h_pooled = torch.cat((h_machines_pooled, h_operations_pooled), dim=-1)
            h_operations_pooled_padding = h_operations_pooled[:, None, None, :].expand_as(h_jobs_padding)
            h_actions = torch.cat((h_machines_padding, h_jobs_padding, h_pairs,
                                   h_machines_pooled_padding, h_operations_pooled_padding), dim=-1)
        elif "job" in self.meta_data[0]:
            h_pooled = torch.cat((h_machines_pooled, h_jobs_pooled), dim=-1)
            h_jobs_pooled_padding = h_jobs_pooled[:, None, None, :].expand_as(h_jobs_padding)
            h_actions = torch.cat((h_machines_padding, h_jobs_padding, h_pairs,
                                   h_machines_pooled_padding, h_jobs_pooled_padding), dim=-1)

        for i in range(self.n_layers_actor):
            if i < len(self.actor) - 1:
                h_actions = self.actor[i](h_actions)
                h_actions = F.elu(h_actions)
            else:
                logits = self.actor[i](h_actions).flatten(1)

        mask_pair = mask_pair.flatten(1)
        logits[~mask_pair] = float('-inf')
        probs = F.softmax(logits, dim=1)

        for i in range(self.n_layers_critic):
            if i < len(self.critic) - 1:
                h_pooled = self.critic[i](h_pooled)
                h_pooled = F.elu(h_pooled)
            else:
                value = self.critic[i](h_pooled)

        return probs, value