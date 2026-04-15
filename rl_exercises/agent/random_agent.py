from rl_exercises.agent import AbstractAgent


class RandomAgent(AbstractAgent):
    def __init__(self, env):
        self.env = env

    def predict_action(self, state, info, evaluate=False):
        return self.env.action_space.sample(), {}
