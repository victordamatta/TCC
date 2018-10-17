import os
import sys
from rlpytorch import *

class Agent:
    def __init__(self, verbose=False, actor_name="actor"):
        ''' Initialization for Trainer. Accepted arguments: ``num_games``, ``batch_size``
            Also need arguments for `Evaluator` and `ModelSaver` class.
        '''
        pass

    def actor(self, batch):
        ''' Actor.
        Get the model, forward the batch and get a distribution. Sample from it and act.
        Reply the message to game engine.

        Args:
            batch(dict): batch data

        Returns:
            reply_msg(dict): ``pi``: policy, ``a``: action, ``V``: value, `rv`: reply version, signatured by step
        '''
        pass

    def train(self, batch):
        ''' Trainer.
        Get the model, forward the batch and update the weights.

        Args:
            batch(dict): batch data
        '''
        pass

    def episode_start(self, i):
        ''' Called before each episode.

        Args:
            i(int): index in the minibatch
        '''
        pass

    def episode_summary(self, i):
        ''' Called after each episode. Print stats and summary. Also print arguments passed in.

        Args:
            i(int): index in the minibatch
        '''
        pass

    def save(self, filename):
        pass
    
    def load(self, filename):
        pass

