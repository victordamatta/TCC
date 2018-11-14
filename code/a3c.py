from agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from itertools import chain
from rlpytorch import add_err, MultiCounter

class A3CAgent(Agent):
    def __init__(self, sampler, stats=None):
        super(A3CAgent, self).__init__()
        self.sampler = sampler
        self.stats = stats
        self.counter = MultiCounter(verbose=False)

        self.net = nn.Sequential(
            nn.Conv2d(22, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 13, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )

        self.pi = nn.Linear(325, 9)
        self.softmax = nn.Softmax()
        self.value = nn.Linear(325, 1)
        
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def actor(self, batch):
        if self.stats is not None:
            self.stats.feed_batch(batch)

        state_curr = batch.hist(0)['s']
        output = self.decide(state_curr)
        return output

    def train(self, batch):
        T = batch["s"].size(0)
        self.optimizer.zero_grad()

        bht = batch.hist(T - 1)
        R = self.decide(bht["s"])["V"].squeeze()
        for i, terminal in enumerate(bht["terminal"]):
            if terminal:
                R[i] = 0.0

        err = None
        verr = None
        perr = None

        for t in range(T - 2, -1, -1):
            bht = batch.hist(t)
            state = self.forward(bht["s"])

            r = batch["r"][t]
            R = self.gamma * R + r
            for i, terminal in enumerate(bht["terminal"]):
                if terminal:
                    R[i] = 0.0

            V = state["V"].squeeze()

            coef = Variable(R - V.data) #.data? -1?
            pi = state["pi"]
            a = bht["a"]

            log_pi = (pi + 1e-6).log()
            def bw_hook(grad_in):
                # this works only on pytorch 0.2.0
                return grad_in.mul(coef.view(-1, 1))
            #log_pi.register_hook(bw_hook)
            log_pi = log_pi.mul(coef.view(-1, 1))

            nlll = nn.NLLLoss()(log_pi, Variable(a))
            mse = nn.MSELoss()(V, Variable(R))
            
            verr = add_err(verr, mse.data[0])
            perr = add_err(perr, nlll.data[0])
            err = add_err(err, mse)
            err = add_err(err, nlll)

        self.counter.stats["vcost"].feed(verr / (T - 1))
        self.counter.stats["pcost"].feed(perr / (T - 1))
        self.counter.stats["cost"].feed(err.data[0] / (T - 1))

        err.backward()
        self.optimizer.step()
        # separate temporary update to permanent update?

    def decide(self, state):
        output = self.forward(state)
        sampled = self.sampler.sample(output)
        sampled['V'] = output['V'].data
        return sampled

    def forward(self, state):
        x = self.net(Variable(state))
        x = x.view(x.size(0), -1)
        output = dict(pi=self.softmax(self.pi(x)), V=self.value(x))
        return output

    def episode_summary(self, i):
        if self.stats is not None:
            self.stats.print_summary()
            if self.stats.count_completed() > 10000:
                self.stats.reset()
        self.counter.summary(global_counter=i)
