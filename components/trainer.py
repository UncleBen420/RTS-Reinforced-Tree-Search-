import os
import random

from tqdm import tqdm

from components.agent import PolicyGradient
from components.environment import Environment


class Trainer:
    def __init__(self):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment()
        self.agent = PolicyGradient(self.env)

    def train(self, nb_episodes, train_path):
        self.img_path = os.path.join(train_path, "img")
        self.label_path = os.path.join(train_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        losses = []
        rewards = []
        vs = []
        td_errors = []
        nb_action = []

        with tqdm(range(nb_episodes), unit="episode") as episode:
            for i in episode:
                # random image selection in the training set
                index = random.randint(0, len(self.img_list) - 1)
                img = os.path.join(self.img_path, self.img_list[index])
                bb = os.path.join(self.label_path, self.label_list[index])

                first_state = self.env.reload_env(img, bb)
                loss, sum_reward, sum_v, sum_tde = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                vs.append(sum_v)
                td_errors.append(sum_tde)
                st = self.env.nb_actions_taken
                nb_action.append(st)

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st, V=sum_v / st, tde=sum_tde)









