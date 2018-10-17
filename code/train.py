from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rlpytorch import *
from random_agent import RandomAgent
import argparse

def load_module(mod):
    ''' Load a python module'''
    sys.path.insert(0, os.path.dirname(mod))
    module = __import__(os.path.basename(mod))
    return module

def load_environment(envs, num_models=None, overrides=dict(), defaults=dict(), **kwargs):
    game = load_module(envs["game"]).Loader()

    env = dict(game=game)
    env.update(kwargs)

    parser = argparse.ArgumentParser()
    all_args = ArgsProvider.Load(parser, env, global_defaults=defaults, global_overrides=overrides)
    return  env, all_args

if __name__ == '__main__':
    stats = Stats("agent")
    runner = SingleProcessRun()
    env, all_args = load_environment(os.environ, runner=runner, stats=stats)
    stats.reset()
    agent = RandomAgent(stats)

    GC = env["game"].initialize()

    GC.reg_callback("train", agent.train)
    GC.reg_callback("actor", agent.actor)
    runner.setup(GC, episode_summary=agent.episode_summary,
                episode_start=agent.episode_start)

    runner.run()

