from __future__ import print_function
import math
import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.utils import clamp_probs
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
import torch.nn.functional as F

def cal_invalid_layers(adj, n_input_nodes):
    ret = []
    for s in adj:
        assert (s.shape[0] + n_input_nodes -1 == s.shape[1])
        tmp = []
        flag = True
        start_time = time.time()
        while (flag):
            flag =False
            for i in range(s.shape[0]):
                if (not i in tmp) and (s[i,:].sum()==0 or (i!=s.shape[0]-1 and s[:,i+n_input_nodes].sum()==0)):
                    # print(i, adj[i, :].sum(), adj[:, i].sum())
                    tmp.append(i)
                    s[i, :] = 0
                    if i!=s.shape[0]-1:
                        s[:, i+n_input_nodes] = 0
                    flag = True
            if time.time() - start_time > .5:
                print(s, 'gggg')
                break
        tmp = np.sort(tmp)
        ret.append(tmp)
    return ret

def sample_concrete_bernoulli(logits, tau, hard=False, eps=1e-10):
    uniforms = clamp_probs(torch.rand(logits.shape, dtype=logits.dtype, device=logits.device))
    ret = ((uniforms.log() - (-uniforms).log1p() + logits) / tau).sigmoid()
    if hard:
        return ((ret >= 0.5).float() - ret).detach() + ret
    else:
        return ret

def sample_sigmoid(y, sample, thresh=0.5, sample_time=2, diff=False):
    '''
        do sampling over unnormalized score
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param sampe_time: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''

    # do sigmoid first
    # y = F.sigmoid(y)
    # do sampling
    if sample:
        if sample_time>1:
            y_result = torch.rand(y.size(0),y.size(1),y.size(2)).cuda()
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(sample_time):
                    y_thresh = torch.rand(y.size(1), y.size(2)).cuda()
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = torch.rand(y.shape).cuda()
            y_result = torch.gt(y,y_thresh).float()
            if diff:
                y_result = (y_result-y).detach() + y
    # do max likelihood based on some threshold
    else:
        y_thresh = (torch.ones(y.shape)*thresh).cuda()
        y_result = torch.gt(y, y_thresh).float()
    return y_result

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
        # self.node_features = [nn.Parameter(1e-3*torch.randn(n_components, n, feature_dim).cuda()) for n in n_nodes]
        # self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
        self.prob_logits = [nn.Parameter(1e-3*torch.randn(n, n).cuda()) for n in n_nodes]
        self.nf_optimizer = torch.optim.Adam(self.prob_logits, lr=adj_lr, betas=(0.5, 0.999))

        self.tau = None
        self.samples = None
        self.counter = 0

        self.fsamples = None

    # straight forward gradient estimator
    def st_sample(self, validate=False):
        # self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
        self.samples, self.invalid_layerss = [], []
        for i, l in enumerate(self.prob_logits):
            if not validate:
                cur_samples = l.sigmoid()
                cur_samples = (torch.bernoulli(cur_samples).float() - cur_samples).detach() + cur_samples
            else:
                cur_samples = (l >= 0.).float()
            invalid_layers = cal_invalid_layers(cur_samples.data.tril(diagonal=-1))
            self.samples.append(cur_samples)
            self.invalid_layerss.append(invalid_layers)
        return self.samples, self.invalid_layerss

    def sample(self, validate=False, ST=False):

        if ST:
            return self.st_sample(validate)

        if not validate:
            self.tau = max(self.tau0 * math.exp(-self.tau_anneal_rate * self.counter), self.tau_min)
            # self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.sample_logits, self.samples, self.invalid_layerss = [], [], []
            for i, l in enumerate(self.prob_logits):
                if self.hard:
                    invalid_layers = [self.n_nodes[i] - 1]
                    while self.n_nodes[i] - 1 in invalid_layers:
                        cur_logits = sample_concrete_bernoulli(l, self.tau)
                        cur_samples = cur_logits.sigmoid()
                        cur_samples = (((cur_samples >= 0.5).float() - cur_samples).detach() + cur_samples)
                        invalid_layers = cal_invalid_layers(cur_samples.data.tril(diagonal=-1))
                        # if cur_samples.shape[0] - 1 in invalid_layers:
                        #     print("----------------", cur_samples.data.tril(diagonal=-1), "----------------")
                else:
                    cur_logits = sample_concrete_bernoulli(l, self.tau)
                    cur_samples = cur_logits.sigmoid()
                    invalid_layers = []
                self.sample_logits.append(cur_logits)
                self.samples.append(cur_samples)
                self.invalid_layerss.append(invalid_layers)
            self.counter += 1
        else:
            # self.prob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.samples, self.invalid_layerss = [], []
            for i, l in enumerate(self.prob_logits):
                if self.hard:
                    cur_samples = (l >= 0.).float()
                    invalid_layers = cal_invalid_layers(cur_samples.data.tril(diagonal=-1))
                else:
                    cur_samples = l.sigmoid()
                    invalid_layers = []
                self.samples.append(cur_samples)
                self.invalid_layerss.append(invalid_layers)
        return self.samples, self.invalid_layerss

    def fixed_adj(self,):
        if self.fsamples is None:
            # self.fprob_logits = [torch.matmul(f[0], f[0].transpose(1,0)) for f in self.node_features]
            self.fsamples, self.finvalid_layerss = [], []
            for i, l in enumerate(self.prob_logits):
                if self.hard:
                    cur_samples = (l >= 0.).float()
                    invalid_layers = cal_invalid_layers(cur_samples.data.tril(diagonal=-1))
                else:
                    cur_samples = l.sigmoid().detach()
                    invalid_layers = []
                self.fsamples.append(cur_samples)
                self.finvalid_layerss.append(invalid_layers)
        return self.fsamples, self.finvalid_layerss

    def neg_entropy(self,):
        ret = 0
        for l in self.prob_logits:
            p = clamp_probs(l.tril(diagonal=-1).sigmoid())
            p.data[1, 0] = 0.5
            p.data[-1, 0] = 0.5
            ret += (p.log()*p + (-p).log1p()*(1-p)).mean()
        return ret

# a deterministic linear output
class MLP_plain(nn.Module):
    def __init__(self, h_size, embedding_size, y_size):
        super(MLP_plain, self).__init__()
        self.deterministic_output = nn.Sequential(
            nn.Linear(h_size, embedding_size),
            # nn.ReLU(),
            # nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, y_size)
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        y = self.deterministic_output(h)
        return y

# plain GRU model
class GRU_plain(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, has_input=True, has_output=False, output_size=None):
        super(GRU_plain, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.has_input = has_input
        self.has_output = has_output

        if has_input:
            self.input = nn.Linear(input_size, embedding_size)
            self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        if has_output:
            self.output = nn.Sequential(
                # nn.Linear(hidden_size, embedding_size),
                # nn.ReLU(),
                # nn.Linear(embedding_size, output_size)
                nn.Linear(hidden_size, output_size),
            )

        self.relu = nn.ReLU()
        # initialize
        self.hidden = None  # need initialize before forward run

        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.25)
            elif 'weight' in name:
                nn.init.xavier_uniform(param,gain=nn.init.calculate_gain('sigmoid'))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = nn.init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()

    def forward(self, input_raw, pack=False, input_len=None):
        if self.has_input:
            input = self.input(input_raw)
            input = self.relu(input)
        else:
            input = input_raw
        if pack:
            input = pack_padded_sequence(input, input_len, batch_first=True)
        output_raw, self.hidden = self.rnn(input, self.hidden)
        if pack:
            output_raw = pad_packed_sequence(output_raw, batch_first=True)[0]
        if self.has_output:
            output_raw = self.output(output_raw)
        # return hidden state at each time step
        return output_raw

class RNNAdjacency(object):
    def __init__(self, n_stages, n_nodes, size="small", lr=0.003, milestones=[10000, 15000], lr_rate=0.3, rl_baseline_decay_weight=0.99, entropy_weight=1., tau0=3., tau_min=1., tau_anneal_rate=3e-5, hard=True):
        super(RNNAdjacency, self).__init__()
        self.n_stages = n_stages
        self.n_nodes = n_nodes # number of nodes to generate
        if 'small' in size:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.max_prev_node = n_nodes - 1
        self.hidden_size_rnn = int(64/self.parameter_shrink) # hidden size for main RNN
        self.hidden_size_rnn_output = 16 # hidden size for output RNN
        self.embedding_size_rnn = int(32/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_rnn_output = 8 # the embedding size for output rnn
        self.num_layers = int(4/self.parameter_shrink)
        self.rnn = GRU_plain(input_size=self.max_prev_node, embedding_size=self.embedding_size_rnn,
                        hidden_size=self.hidden_size_rnn, num_layers=self.num_layers, has_input=True,
                        has_output=True, output_size=self.hidden_size_rnn_output).cuda()
        self.output = GRU_plain(input_size=1, embedding_size=self.embedding_size_rnn_output,
                           hidden_size=self.hidden_size_rnn_output, num_layers=self.num_layers, has_input=True,
                           has_output=True, output_size=1).cuda()

        # initialize optimizer
        self.lr = lr
        self.milestones = milestones
        self.lr_rate = lr_rate
        self.optimizer_rnn = torch.optim.Adam(list(self.rnn.parameters()), lr=self.lr)
        self.optimizer_output = torch.optim.Adam(list(self.output.parameters()), lr=self.lr)
        self.scheduler_rnn = MultiStepLR(self.optimizer_rnn, milestones=self.milestones, gamma=self.lr_rate)
        self.scheduler_output = MultiStepLR(self.optimizer_output, milestones=self.milestones, gamma=self.lr_rate)

        self.baseline = None
        self.baseline_decay_weight = rl_baseline_decay_weight

        self.entropy_weight = entropy_weight

        # temperature for gsm
        self.counter = 0
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        self.hard = hard

    def sample(self, batch_size=1, deterministic=False):

        # self.tau = max(self.tau0 * math.exp(-self.tau_anneal_rate * self.counter), self.tau_min)

        self.rnn.hidden = self.rnn.init_hidden(batch_size)
        batch_adj = torch.zeros(batch_size, self.n_nodes*self.n_stages, self.max_prev_node, requires_grad=False).cuda()
        self.neg_ent = 0.
        self.logp = 0.
        for ite in range(self.n_nodes*self.n_stages):
            i = ite % self.n_nodes
            if i == 0:
                x_step = torch.ones(batch_size,1,self.max_prev_node).cuda()
                continue
            h = self.rnn(x_step)
            # output.hidden = h.permute(1,0,2)
            hidden_null = torch.zeros(self.num_layers - 1, h.size(0), h.size(2)).cuda()
            self.output.hidden = torch.cat((h.permute(1,0,2), hidden_null),
                                      dim=0)  # num_layers, batch_size, hidden_size
            x_step = torch.zeros(batch_size,1,self.max_prev_node).cuda()
            output_x_step = torch.ones(batch_size,1,1).cuda()
            for j in range(i):
                output_y_pred_step = self.output(output_x_step)
                p = clamp_probs(output_y_pred_step.sigmoid())
                output_x_step = sample_sigmoid(p, sample=not deterministic, sample_time=1)
                self.logp += (p.log()*output_x_step + (-p).log1p()*(1-output_x_step)).sum(dim=(1,2))
                self.neg_ent += (p.log()*p + (-p).log1p()*(1-p)).sum(dim=(1,2))
                x_step[:,:,j:j+1] = output_x_step
                # self.output.hidden = self.output.hidden.data#.cuda()
            batch_adj[:, ite:ite+1, :] = x_step
            # self.rnn.hidden = self.rnn.hidden.data#.cuda()
        return batch_adj

    def zero_grad(self,):
        self.rnn.zero_grad()
        self.output.zero_grad()

    def step(self, rewards, epoch):
        self.avg_reward = rewards.mean()
        if self.baseline is None:
            self.baseline = self.avg_reward
        else:
            self.baseline += self.baseline_decay_weight * (self.avg_reward - self.baseline)

        self.neg_entropy = self.neg_ent.mean()
        self.rl_loss = (-(rewards - self.avg_reward)/(rewards.std() + 1e-12) * self.logp).mean()
        loss = self.rl_loss + self.neg_entropy * self.entropy_weight*max(1-float(epoch)/160., 0)# * math.pow(0.973, epoch)
        (loss*0.1).backward()

        nn.utils.clip_grad_norm_(self.rnn.parameters(), 0.25)
        nn.utils.clip_grad_norm_(self.output.parameters(), 0.25)
        self.optimizer_output.step()
        self.optimizer_rnn.step()
        self.scheduler_output.step()
        self.scheduler_rnn.step()
        self.counter += 1

    def train(self,):
        self.rnn.train()
        self.output.train()

    def eval(self,):
        self.rnn.eval()
        self.output.eval()

    def save(self, path, epoch):
        torch.save(self.rnn.state_dict(), path + 'rnn_' + str(epoch) + '.dat')
        torch.save(self.output.state_dict(), path + 'output_' + str(epoch) + '.dat')

    def load(self, path=None, epoch = 0):
        if not path:
            path = 'snapshots/pretrain_adj/'
        self.rnn.load_state_dict(torch.load(path + 'rnn_' + str(epoch) + '.dat'))
        self.output.load_state_dict(torch.load(path + 'output_' + str(epoch) + '.dat'))


class RNNMLPAdjacency(object):
    def __init__(self, n_stages, n_nodes, lr=0.003, milestones=[10000, 15000], lr_rate=0.3, entropy_weight=1., tau0=3., tau_min=1., tau_anneal_rate=5e-4, hard=False):
        super(RNNMLPAdjacency, self).__init__()
        self.n_stages = n_stages
        self.n_nodes = n_nodes # number of nodes to generate
        if n_nodes <= 18:
            self.parameter_shrink = 2
        else:
            self.parameter_shrink = 1
        self.max_prev_node = n_nodes - 1
        self.hidden_size_rnn = int(128/self.parameter_shrink) # hidden size for main RNN
        self.embedding_size_rnn = int(64/self.parameter_shrink) # the size for LSTM input
        self.embedding_size_output = int(128/self.parameter_shrink)
        self.num_layers = 2#int(4/self.parameter_shrink)

        self.rnn = GRU_plain(input_size=self.max_prev_node+self.n_stages, embedding_size=self.embedding_size_rnn,
                        hidden_size=self.hidden_size_rnn, num_layers=self.num_layers, has_input=True,
                        has_output=False).cuda()
        self.output = MLP_plain(h_size=self.hidden_size_rnn, embedding_size=self.embedding_size_output,
                        y_size=self.max_prev_node).cuda()

        # initialize optimizer
        self.lr = lr
        self.milestones = milestones
        self.lr_rate = lr_rate
        self.optimizer = torch.optim.Adam(list(self.rnn.parameters())+list(self.output.parameters()), lr=self.lr)
        # self.optimizer_output = torch.optim.Adam(, lr=self.lr)
        self.scheduler = MultiStepLR(self.optimizer, milestones=self.milestones, gamma=self.lr_rate)
        # self.scheduler_output = MultiStepLR(self.optimizer_output, milestones=self.milestones, gamma=self.lr_rate)

        self.entropy_weight = entropy_weight
        self.stage_m = torch.eye(self.n_stages).cuda().view(self.n_stages, 1, self.n_stages)

        # temperature for gsm
        self.counter = 0
        self.tau0 = tau0
        self.tau_min = tau_min
        self.tau_anneal_rate = tau_anneal_rate
        self.hard = hard

    def sample(self, batch_size=1, deterministic=False):

        self.rnn.hidden = self.rnn.init_hidden(batch_size*self.n_stages)
        batch_adj = torch.zeros(batch_size*self.n_stages, self.n_nodes, self.max_prev_node, requires_grad=False).cuda()
        self.neg_ent = 0.
        self.logp = 0.
        for ite in range(self.n_nodes):
            i = ite % self.n_nodes
            if i == 0:
                x_step = torch.ones(batch_size*self.n_stages,1,self.max_prev_node).cuda()
                continue
            h = self.rnn(torch.cat([self.stage_m.repeat(batch_size, 1, 1), x_step], 2))
            y_pred_step = self.output(h)
            p = clamp_probs(F.sigmoid(y_pred_step))
            x_step = sample_sigmoid(p, sample=not deterministic, sample_time=1)
            x_step[:,:,i:] = 0
            self.logp += (p.log()*x_step + (-p).log1p()*(1-x_step))[:,:,:i].sum(dim=(1,2))
            self.neg_ent += (p.log()*p + (-p).log1p()*(1-p))[:,:,:i].sum(dim=(1,2))
            batch_adj[:, ite:ite+1, :] = x_step
            # rnn.hidden = Variable(rnn.hidden.data).cuda()
        self.logp = self.logp.view(batch_size, self.n_stages).sum(1)
        self.neg_ent = self.neg_ent.view(batch_size, self.n_stages).sum(1)
        batch_adj = batch_adj.view(batch_size, self.n_stages, self.n_nodes, self.max_prev_node).view(batch_size, -1, self.max_prev_node)
        return batch_adj

    def sample_gs(self, batch_size=1, deterministic=False):

        self.tau = max(self.tau0 * math.exp(-self.tau_anneal_rate * self.counter), self.tau_min)

        self.rnn.hidden = self.rnn.init_hidden(batch_size*self.n_stages)
        batch_adj = torch.zeros(batch_size*self.n_stages, self.n_nodes, self.max_prev_node, requires_grad=False).cuda()
        self.neg_ent = 0.
        # self.logp = 0.
        for ite in range(self.n_nodes):
            i = ite % self.n_nodes
            if i == 0:
                x_step = torch.ones(batch_size*self.n_stages,1,self.max_prev_node).cuda()
                continue
            h = self.rnn(torch.cat([self.stage_m.repeat(batch_size, 1, 1), x_step], 2))
            y_pred_step = self.output(h)[:,:,:i]
            p = y_pred_step.sigmoid()
            batch_adj[:, ite:ite+1, :i] = sample_concrete_bernoulli(y_pred_step, self.tau, self.hard) if not deterministic else sample_sigmoid(p,False,sample_time=1)
            # self.logp += (p.log()*x_step + (-p).log1p()*(1-x_step))[:,:,:i].sum(dim=(1,2))
            self.neg_ent += (-F.softplus(-y_pred_step)*p - F.softplus(y_pred_step)*(1-p)).sum(dim=(1,2))
            x_step = batch_adj[:, ite:ite+1, :].data
            # rnn.hidden = Variable(rnn.hidden.data).cuda()
        # self.logp = self.logp.view(batch_size, self.n_stages).sum(1)
        self.neg_ent = self.neg_ent.view(batch_size, self.n_stages).sum(1).mean() * self.entropy_weight
        batch_adj = batch_adj.view(batch_size, self.n_stages, self.n_nodes, self.max_prev_node).view(batch_size, -1, self.max_prev_node)
        return batch_adj

    def zero_grad(self,):
        self.optimizer.zero_grad()

    def step(self, ):
        nn.utils.clip_grad_norm_(self.rnn.parameters(), 0.25)
        self.optimizer.step()
        self.scheduler.step()
        self.counter += 1

    def rl_step(self, rewards, epoch):
        self.avg_reward = rewards.mean()
        if self.baseline is None:
            self.baseline = self.avg_reward
        else:
            self.baseline += self.baseline_decay_weight * (self.avg_reward - self.baseline)

        self.neg_entropy = self.neg_ent.mean()
        self.rl_loss = (-(rewards - self.avg_reward)/(rewards.std() + 1e-12) * self.logp).mean()
        loss = self.rl_loss + self.neg_entropy * self.entropy_weight*max(1-float(epoch)/160., 0)# * math.pow(0.973, epoch)
        (loss).backward()
        self.step()

    def train(self,):
        self.rnn.train()
        self.output.train()

    def eval(self,):
        self.rnn.eval()
        self.output.eval()

    def save(self, path, epoch):
        torch.save(self.rnn.state_dict(), path + 'rnn_' + str(epoch) + '.dat')
        torch.save(self.output.state_dict(), path + 'output_' + str(epoch) + '.dat')

    def load(self, path=None, epoch = 0):
        if not path:
            path = 'snapshots/pretrain_rnnmlpadj/'
        self.rnn.load_state_dict(torch.load(path + 'rnn_' + str(epoch) + '.dat'))
        self.output.load_state_dict(torch.load(path + 'output_' + str(epoch) + '.dat'))


if __name__ == "__main__":
    N = 34
    M = 3

    available_adjs = []
    available_adjs.append(torch.diag(torch.ones(N-1)).cuda())
    available_adjs.append(torch.diag(torch.ones(N-2), -1).cuda())
    available_adjs.append(torch.diag(torch.ones(N-3), -2).cuda())
    available_adjs.append(available_adjs[-1])
    # available_adjs.append(torch.diag(torch.ones(2), -4).cuda())
    # available_adjs.append(torch.diag(torch.ones(1), -5).cuda())

    adj = RNNMLPAdjacency(M, N, entropy_weight=0.03, milestones=[2000, 3000], hard=False, tau0=3., tau_min=1., tau_anneal_rate=math.log(3./1.)/6000.)
    print('  + Number of params of rnn: {}'.format(sum([p.data.nelement() for p in adj.rnn.parameters()])))
    print('  + Number of params of output: {}'.format(sum([p.data.nelement() for p in adj.output.parameters()])))
    adj.train()
    for i in range(3000):
        adj.zero_grad()
        samp = adj.sample_gs(1)
        rewards = []
        for j in range(samp.shape[0]):
            tmp = 0.
            samps = samp[j].chunk(M)
            for k in range(M):
                # tmp_ = []
                # for h in available_adjs:
                #     tmp_.append((h==k[1:]).float().tril().sum()/21.)
                tmp += ((available_adjs[2-k]-samps[k][1:])**2).tril().sum()/3./N/(N-1)*2.
            rewards.append(tmp)
        loss = sum(rewards) + adj.neg_ent*max(1-float(i)/1000., 0)
        loss.backward()
        adj.step()
        # adj.step(torch.from_numpy(np.array(rewards)).float().cuda(), i/5.)
        print(i, sum(rewards).item(), adj.tau)
        # if i % 30 == 29:
        #     adj.scheduler.step()

    adj.eval()
    with torch.no_grad():
        samp = adj.sample_gs(1, deterministic=True)
    for s in samp[0]:
        print(s.data.cpu().numpy())
    adj.save('snapshots/pretrain_adj/', 0)
