import random
import time

import numpy as np
from tqdm import tqdm


class DummyAgent:

    def __init__(self, environment):

        self.environment = environment
        self.action_space = 4

    def exploit_one_episode(self):
        existing_proba = None
        while True:
            # casting to torch tensor

            if existing_proba is None:
                probs = np.random.rand(self.action_space)
            else:
                probs = existing_proba

            # no need to explore, so we select the most probable action
            A, Q = self.environment.exploit(probs)
            _, _, is_terminal, _, existing_pred = self.environment.take_action(A, Q)

            existing_proba = existing_pred

            if is_terminal:
                break

