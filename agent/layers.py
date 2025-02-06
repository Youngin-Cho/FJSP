import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipConnection(nn.Module):
    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input):
        return input + self.module(input)


class Normalization(nn.Module):
    def __init__(self, hidden_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(hidden_dim, affine=True)
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class FeedForward(nn.Module):
    def __init__(self, n_layers, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()

        self.n_layers = n_layers
        self.layers = torch.nn.ModuleList()

        if n_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(n_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        for l in range(self.n_layers - 1):
            input = F.relu(self.layers[l](input))
        return self.layers[self.n_layers - 1](input)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, input_dim, embed_dim):
        super(MultiHeadAttention, self).__init__()

        val_dim = embed_dim // n_heads
        key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        self.W_out = nn.Parameter(torch.Tensor(n_heads, val_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h=None, mask=None):
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, n_entity, input_dim)
        batch_size, n_entity, input_dim = h.size()
        n_query = q.size(1)

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, n_entity, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, batch_size, n_query, key/val_dim)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, n_entity, key/val_dim)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, n_entity)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))

        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, 1, n_entity).expand_as(compatibility)
            compatibility[mask] = -np.inf

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        # Calculate compatibility (n_heads, batch_size, n_query, val_dim)
        heads = torch.matmul(attn, V)

        # Calculate output (batch_size, n_query, embed_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        return out


class Transformer(nn.Sequential):
    def __init__(self,n_heads, input_dim, embed_dim, n_layers_ff=2, hidden_dim_ff=512, normalization='batch'):
        super(Transformer, self).__init__()

        self.mha = MultiHeadAttention(n_heads, input_dim=input_dim, embed_dim=embed_dim)
        self.residual1 = SkipConnection(self.mha)
        self.norm1 = Normalization(embed_dim, normalization)

        self.ffn = FeedForward(n_layers_ff, input_dim=embed_dim, hidden_dim=hidden_dim_ff, output_dim=embed_dim)
        self.residual2 = SkipConnection(self.ffn)
        self.norm2 = Normalization(embed_dim, normalization)

    def forward(self, q, h=None, mask=None):
        out = self.mha(q, h=h, mask=mask)
        out = self.residual1(out)
        out = self.norm1(out)

        out = self.ffn(out)
        out = self.residual2(out)
        out = self.norm2(out)

        return out