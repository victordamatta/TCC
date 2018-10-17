from agent import Agent
import torch

class LinearAgent(Agent):
    def __init__(self, stats=None):
        super(LinearAgent, self).__init__()
        self.stats = stats

    def actor(self, batch):
        if self.stats is not None:
            self.stats.feed_batch(batch)
        reply = dict()
        reply["pi"] = torch.ones([batch.batchsize, 9]) / 9.0
        reply["V"] = torch.zeros([batch.batchsize])
        reply["a"] = torch.IntTensor(batch.batchsize).random_(1, 9)
        return reply

    def episode_summary(self, i):
        if self.stats is not None:
            self.stats.print_summary()
            if self.stats.count_completed() > 10000:
                self.stats.reset()