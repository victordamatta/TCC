from agent import Agent
import torch
from rlpytorch import Stats

class RandomAgent(Agent):
    def __init__(self, stats=None):
        super(RandomAgent, self).__init__()
        self.stats = stats

    def actor(self, batch):
        #print("actor: ", batch.batch.keys())
        if self.stats is not None:
            self.stats.feed_batch(batch)
        reply = dict()
        reply["pi"] = torch.ones([batch.batchsize, 9]) / 9.0
        reply["V"] = torch.zeros([batch.batchsize])
        reply["a"] = torch.IntTensor(batch.batchsize).random_(1, 9)
        return reply

    def train(self, batch):
        #print("train: ", batch.batch.keys())
        T = batch["s"].size(0)
        bht = batch.hist(T - 1)
        r = bht["r"]
        last_r = bht["last_r"]
        for i, terminal in enumerate(bht["terminal"]):
            if terminal:
                print("terminal reward: ", r[i], "pre-terminal: ", last_r[i])

    def episode_summary(self, i):
        if self.stats is not None:
            self.stats.print_summary()
            if self.stats.count_completed() > 10000:
                self.stats.reset()