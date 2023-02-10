import os
import random
import numpy as np
from tqdm import tqdm
from components.agent import PolicyGradient
from components.dummy_agent import DummyAgent
from components.environment import Environment
from components.plot import MetricMonitor, metrics_to_pdf, metrics_eval_to_pdf


def describe(arr):
    """
    show several metrics from a given array.
    @param arr: array to plot the metrics from.
    """
    print("Measures of Central Tendency")
    print("Mean =", np.mean(arr))
    print("Median =", np.median(arr))
    print("Measures of Dispersion")
    print("Minimum =", np.min(arr))
    print("Maximum =", np.max(arr))
    print("Variance =", np.var(arr))
    print("Standard Deviation =", np.std(arr))


class Trainer:
    """
    This class act has a middle ground between the environment ant the agent. It use the unload the agent to the task
    of keeping tracks of the metrics.
    """
    def __init__(self, epsilon, learning_rate, gamma, lr_gamma, min_res):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment(epsilon=epsilon, res_min=min_res)
        self.agent = PolicyGradient(self.env, learning_rate=learning_rate, gamma=gamma, lr_gamma=lr_gamma)
        self.dummy = DummyAgent(self.env)

    def train(self, nb_episodes, train_path, result_path='.',
              real_time_monitor=False, plot_metric=False, transfer_learning=False):
        """
        Allow user to train RTS.
        @param nb_episodes: number of episode needed to train the agent.
        @param train_path: path where the training data is.
        @param result_path: path where the runs will be saved.
        @param real_time_monitor: if True, run a server showing realtime progression of the training.
        @param plot_metric: if True return also metric in .pdf format in the runs folder.
        @param transfer_learning: use already existing weights.
        """

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING PREPARATION
        # --------------------------------------------------------------------------------------------------------------
        if transfer_learning:
            self.agent.load(transfer_learning)

        self.agent.model_summary() # show the QNet summary.

        # create the monitor server.
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

                if real_time_monitor:
                    metric_monitor.update_values(sum_reward, loss, st)

                episode.set_postfix(loss=loss, nb_action=st)

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
        self.agent.save(os.path.join(path_weights, "weights_rts.pt"))

        if plot_metric:
            path_plot = os.path.join(path, "plot/")
            try:
                os.mkdir(path_plot)
            except OSError as error:
                print(error)
            metrics_to_pdf(rewards, losses,
                           nb_action, nb_conv_action, nb_min_zoom_action,
                           path_plot, "training")

    def evaluate(self, eval_path, result_path='.', plot_metric=False):
        """
        Evaluate RTS on the testing set.
        @param eval_path: path where the testing data is kept.
        @param result_path: path where the runs will be saved.
        @param plot_metric: if True return also metric in .pdf format in the runs folder.
        """

        self.img_path = os.path.join(eval_path, "img")
        self.label_path = os.path.join(eval_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        rewards = []
        good_hits_ratio = []
        nb_action = []
        nb_action_min = []
        nb_conv_action = []
        nb_action_dummy = []
        nb_action_min_dummy = []
        precision = []
        pertinence = []
        count_per_action = []
        hist_agent = []
        hist_conv = []
        hist_dummy = []

        if plot_metric:
            path = os.path.join(result_path, "rts_runs/evaluation")
            path_plot = os.path.join(path, "plot/")
            try:
                os.mkdir(path)
                os.mkdir(path_plot)
            except OSError as error:
                print(error)

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
                stm = self.env.min_zoom_action

                rewards.append(sum_reward)
                nb_action.append(st)
                nb_action_min.append(stm)
                nb_conv_action.append(self.env.conventional_policy_nb_step)
                count_per_action.append(self.env.count_per_action)

                pr = self.env.min_zoom_action / self.env.nb_max_conv_action
                prc = self.env.conventional_policy_nb_step / self.env.nb_max_conv_action
                precision.append(1 - pr)
                pertinence.append(prc - pr)

                good_hits = self.env.good_hits / st
                good_hits_ratio.append(good_hits)

                hist_agent.extend((self.env.marked / 10).astype(int).tolist())
                hist_conv.extend((self.env.conventional_policy_nb_step_per_object / 10).astype(int).tolist())

                if i % 3 == 0 and plot_metric:
                    self.env.get_gif_trajectory(os.path.join(result_path, "rts_runs/evaluation", img_filename + ".gif"))

                episode.set_postfix(rewards=sum_reward, nb_action=st, good_hits_ratio=good_hits)

        with tqdm(range(len(self.img_list)), unit="episode") as episode:
            for i in episode:
                img_filename = self.img_list[i]
                img = os.path.join(self.img_path, img_filename)
                bb = os.path.join(self.label_path, img_filename[:-4] + '.txt')
                if not os.path.exists(bb):
                    continue

                self.env.reload_env(img, bb)
                self.dummy.exploit_one_episode()
                st = self.env.nb_actions_taken
                stm = self.env.min_zoom_action
                nb_action_dummy.append(st)
                nb_action_min_dummy.append(stm)

                hist_dummy.extend((self.env.marked / 10).astype(int).tolist())

                episode.set_postfix(nb_action=st)

        # --------------------------------------------------------------------------------------------------------------
        # PLOT
        # --------------------------------------------------------------------------------------------------------------
        print("Steps taken")
        print("-------------------------------")
        describe(nb_action)
        print("-------------------------------\n")

        print("Means per actions")
        count_per_action = np.array(count_per_action)
        count_per_action_mean = np.mean(count_per_action, axis=0)
        print(count_per_action_mean)
        print("-------------------------------")
        describe(count_per_action_mean)
        print("-------------------------------\n")

        print("Overall precision on evaluation")
        print("-------------------------------")
        describe(precision)
        print("-------------------------------\n")

        print("Overall pertinence on evaluation")
        print("-------------------------------")
        describe(pertinence)
        print("-------------------------------")

        if plot_metric:
            metrics_eval_to_pdf(good_hits_ratio, rewards, nb_action, nb_action_min, nb_conv_action, nb_action_dummy,
                                nb_action_min_dummy, pertinence, precision, hist_agent, hist_conv, hist_dummy,
                                path_plot, "evaluation")
