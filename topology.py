import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.utils import clamp_probs

def cal_io_layers(adj):
    tmp = []
    flag = True
    start_time = time.time()
    while (flag):
        flag =False
        for i in range(2, adj.shape[0] - 1):
            if (not i in tmp) and (adj[i, :].sum() == 0 or adj[:, i].sum() == 0):
                # print(i, adj[i, :].sum(), adj[:, i].sum())
                tmp.append(i)
                adj[i, :] = 0
                adj[:, i] = 0
                flag = True
        if time.time() - start_time > .5:
            print(adj, 'gggg')
            break
    tmp = np.sort(tmp)
    return tmp

def sample_concrete_bernoulli(logits, tau, eps=1e-10):
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    uniforms = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))
    return (uniforms.log() - (-uniforms).log1p() + logits) / tau

class Adjacency(object):
    def __init__(self, n_nodes, n_components, feature_dim, tau0, tau_min, tau_anneal_rate, hard=True, gradient_estimator='gsm', adj_lr=3e-4):
        assert n_components == 1
        assert gradient_estimator == 'gsm'
        self.n_nodes = n_nodes
        self.n_components = n_components
        self.feature_dim = feature_dim
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        self.hard = hard
        self.gradient_estimator = gradient_estimator
        self.adj_lr = adj_lr

        # self.component_logits = [nn.Parameter(torch.zero(n_components).cuda()) for _ in n_nodes]
        self.node_features = [nn.Parameter(1e-3*torch.randn(n_components, n, feature_dim).cuda()) for n in n_nodes]
        self.nf_optimizer = torch.optim.Adam(self.node_features, lr=adj_lr)

        self.tau = None
        self.samples = None
        self.counter = 0

        self.fsamples = None

    def sample(self, validate=False):

        if not validate:
            self.tau = max(self.tau0 * math.exp(-self.tau_anneal_rate * self.counter), self.tau_min)
            self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.sample_logits, self.samples, self.invalid_layerss = [], [], []
            for i, l in enumerate(self.prob_logits):
                if self.hard:
                    cur_logits = sample_concrete_bernoulli(l, self.tau)
                    cur_samples = cur_logits.sigmoid()
                    cur_samples = (((cur_samples >= 0.5).float() - cur_samples).detach() + cur_samples)
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = cal_invalid_layers(cur_samples.data.clone().tril(diagonal=-1))
                else:
                    cur_logits = sample_concrete_bernoulli(l, self.tau)
                    cur_samples = cur_logits.sigmoid()
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = []
                self.sample_logits.append(cur_logits)
                self.samples.append(cur_samples)
                self.invalid_layerss.append(invalid_layers)
            self.counter += 1
        else:
            self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.samples, self.invalid_layerss = [], []
            for i, l in enumerate(self.prob_logits):
                if self.hard:
                    cur_samples = (l >= 0.).float()
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = cal_invalid_layers(cur_samples.data.clone().tril(diagonal=-1))
                else:
                    cur_samples = l.sigmoid()
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = []
                self.samples.append(cur_samples)
                self.invalid_layerss.append(invalid_layers)
        return self.samples, self.invalid_layerss

    def fixed_adj(self,):
        if self.fsamples is None:
            self.fprob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.fsamples, self.finvalid_layerss = [], []
            for i, l in enumerate(self.fprob_logits):
                if self.hard:
                    cur_samples = (l >= 0.).float()
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = cal_invalid_layers(cur_samples.data.clone().tril(diagonal=-1))
                else:
                    cur_samples = l.sigmoid().detach()
                    cur_samples.data[-1, 0] = 1.
                    invalid_layers = []
                self.fsamples.append(cur_samples)
                self.finvalid_layerss.append(invalid_layers)
        return self.fsamples, self.finvalid_layerss

    def neg_entropy(self,):
        ret = 0
        for l in self.prob_logits:
            p = clamp_probs(l.tril(diagonal=-1).sigmoid())
            p.data[1, 0] = 0.5
            ret += (p.log()*p + (-p).log1p()*(1-p)).mean()
        return ret
