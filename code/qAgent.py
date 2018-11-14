from agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class qAgent(Agent):
    def __init__(self, sampler, stats=None):
        super(qAgent, self).__init__()
        self.sampler = sampler
        self.stats = stats
        self.model = nn.Sequential(
          nn.Linear(22*20*20 + 9, 100),
          nn.ReLU(),
          nn.Linear(100,100),
          nn.ReLU(),
          nn.Linear(100, 1)
        )

    def actor(self, batch):
        if self.stats is not None:
            self.stats.feed_batch(batch)

        state_curr = batch.hist(0)['s']

        output = self.forward(state_curr)

        return output

    def train(self, batch):
        T = batch["s"].size(0)

        state_curr = batch.hist(T - 1)['s']
        output = self.forward(state_curr)

        R = output["V"]

        err = None

        for t in range(T - 2, -1, -1):
            bht = batch.hist(t)
            output = self.forward(bht["s"])

            # go through the sample and get the rewards.
            V = output["V"]

            r = bht["last_r"]
            R = self.gamma * R + r
            for i, terminal in enumerate(bht["terminal"]):
                if terminal:
                    R[i] = r[i]

            coef = R - V.data
            pi = output["pi"]
            old_pi = bht["pi"]
            a = bht["a"]

            log_pi = -1 * nn.NLLLoss(pi.log(), Variable(a))

            policy_err = self.pg.feed(R-V.data, state_curr, bht, stats, old_pi_s=bht)
            err = add_err(err, policy_err)
            err = add_err(err, self.value_matcher.feed({ value_node: V, "target" : R}, stats))

        err.backward()

    def forward(self, state, action):
        dims = state.size()
        state = Variable(state.view((dims[0], -1)))

        a = torch.zeros(9).scatter_(0, torch.tensor([action - 1]), 1)

        return self.model(state.cat_(a))


    def episode_summary(self, i):
        if self.stats is not None:
            self.stats.print_summary()
            if self.stats.count_completed() > 10000:
                self.stats.reset()