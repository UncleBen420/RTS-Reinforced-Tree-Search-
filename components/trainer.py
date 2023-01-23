import gc
import os
import random

import numpy as np
from tqdm import tqdm

from components.agent import PolicyGradient
from components.environment import Environment
from components.plot import MetricMonitor, metrics_to_pdf, metrics_eval_to_pdf


def describe(arr):
    print("Measures of Central Tendency")
    print("Mean =", np.mean(arr))
    print("Median =", np.median(arr))
    print("Measures of Dispersion")
    print("Minimum =", np.min(arr))
    print("Maximum =", np.max(arr))
    print("Variance =", np.var(arr))
    print("Standard Deviation =", np.std(arr))


class Trainer:
    def __init__(self, epsilon, learning_rate, gamma, lr_gamma):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment(epsilon=epsilon)
        self.agent = PolicyGradient(self.env, learning_rate=learning_rate, gamma=gamma, lr_gamma=lr_gamma)

    def train(self, nb_episodes, train_path, result_path='.',
              real_time_monitor=False, plot_metric=False, transfer_learning=False):

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING PREPARATION
        # --------------------------------------------------------------------------------------------------------------
        if transfer_learning:
            self.agent.load(transfer_learning)

        self.agent.model_summary()

        if real_time_monitor:
            print("[INFO] Real time monitor server running on localhost.")
            metric_monitor = MetricMonitor()
            metric_monitor.start_server()

        self.img_path = os.path.join(train_path, "img")
        self.label_path = os.path.join(train_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        losses = []
        rewards = []
        good_hits_ratio = []
        nb_action = []
        nb_conv_action = []
        nb_min_zoom_action = []

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(nb_episodes), unit="episode") as episode:
            for i in episode:
                # random image selection in the training set
                while True:
                    index = random.randint(0, len(self.img_list) - 1)
                    img = os.path.join(self.img_path, self.img_list[index])
                    bb = os.path.join(self.label_path, self.img_list[index][:-4] + '.txt')
                    if os.path.exists(bb):
                        break

                first_state = self.env.reload_env(img, bb)
                loss, sum_reward = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                st = self.env.nb_actions_taken
                nb_action.append(st)
                nb_conv_action.append(self.env.conventional_policy_nb_step)
                nb_min_zoom_action.append(self.env.min_zoom_action)

                good_hits = self.env.good_hits / st
                good_hits_ratio.append(good_hits)

                if real_time_monitor:
                    metric_monitor.update_values(st, sum_reward, loss , st)

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st, good_hits_ratio=good_hits)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT AND WEIGHTS SAVING
        # --------------------------------------------------------------------------------------------------------------
        if real_time_monitor:
            metric_monitor.stop_server()
            print("[INFO] Real time monitor server has been stopped")

        path = os.path.join(result_path, "rts_runs")
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        path = os.path.join(path, "training")
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)

        path_weights = os.path.join(path, "weights")
        try:
            os.mkdir(path_weights)
        except OSError as error:
            print(error)
        # saving the model weights
        self.agent.save(os.path.join(path_weights, "weights_rts.pth"))

        if plot_metric:
            path_plot = os.path.join(path, "plot/")
            try:
                os.mkdir(path_plot)
            except OSError as error:
                print(error)
            metrics_to_pdf(good_hits_ratio, rewards, losses,
                           nb_action, nb_conv_action, nb_min_zoom_action,
                           path_plot, "training")

    def evaluate(self, eval_path, result_path='.', plot_metric=False):

        self.img_path = os.path.join(eval_path, "img")
        self.label_path = os.path.join(eval_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        rewards = []
        good_hits_ratio = []
        nb_action = []
        nb_conv_action = []
        precision = []
        pertinence = []

        # --------------------------------------------------------------------------------------------------------------
        # EVALUATION STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(len(self.img_list)), unit="episode") as episode:
            for i in episode:
                img_filename = self.img_list[i]
                img = os.path.join(self.img_path, img_filename)
                bb = os.path.join(self.label_path, img_filename[:-4] + '.txt')
                if not os.path.exists(bb):
                    continue

                first_state = self.env.reload_env(img, bb)
                sum_reward = self.agent.exploit_one_episode(first_state)
                st = self.env.nb_actions_taken
                rewards.append(sum_reward)
                nb_action.append(st)
                nb_conv_action.append(self.env.conventional_policy_nb_step)

                pr = self.env.min_zoom_action / self.env.nb_max_conv_action
                prc = self.env.conventional_policy_nb_step / self.env.nb_max_conv_action
                precision.append(1 - pr)
                pertinence.append(prc - pr)

                good_hits = self.env.good_hits / st
                good_hits_ratio.append(good_hits)

                episode.set_postfix(rewards=sum_reward, nb_action=st, good_hits_ratio=good_hits)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------
        print("")
        print("Overall precision on evaluation")
        print("-------------------------------")
        describe(precision)
        print("-------------------------------")

        print("")
        print("Overall pertinence on evaluation")
        print("-------------------------------")
        describe(pertinence)
        print("-------------------------------")

        if plot_metric:
            path = os.path.join(result_path, "rts_runs/evaluation")
            path_plot = os.path.join(path, "plot/")
            try:
                os.mkdir(path)
                os.mkdir(path_plot)
            except OSError as error:
                print(error)

            metrics_eval_to_pdf(good_hits_ratio, rewards, nb_action, nb_conv_action,
                                pertinence, precision, path_plot, "evaluation")
