import os
import random

from tqdm import tqdm

from components.agent import PolicyGradient
from components.environment import Environment
from components.plot import MetricMonitor, metrics_to_pdf, metrics_eval_to_pdf


class Trainer:
    def __init__(self):
        self.label_path = None
        self.img_path = None
        self.label_list = None
        self.img_list = None
        self.env = Environment()
        self.agent = PolicyGradient(self.env)

    def train(self, nb_episodes, train_path, result_path='.',
              real_time_monitor=False, plot_metric=False, transfer_learning=False):

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING PREPARATION
        # --------------------------------------------------------------------------------------------------------------
        if transfer_learning:
            self.agent.load(transfer_learning)

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
        vs = []
        td_errors = []
        nb_action = []
        nb_conv_action = []

        # --------------------------------------------------------------------------------------------------------------
        # LEARNING STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(nb_episodes), unit="episode") as episode:
            for i in episode:
                # random image selection in the training set
                index = random.randint(0, len(self.img_list) - 1)
                img = os.path.join(self.img_path, self.img_list[index])

                bb = os.path.join(self.label_path, self.img_list[index].split('.')[0] + '.txt')

                first_state = self.env.reload_env(img, bb)
                loss, sum_reward, sum_v, mean_tde = self.agent.fit_one_episode(first_state)

                rewards.append(sum_reward)
                losses.append(loss)
                vs.append(sum_v)
                td_errors.append(mean_tde)
                st = self.env.nb_actions_taken
                nb_action.append(st)
                nb_conv_action.append(self.env.conventional_policy_nb_step)

                if real_time_monitor:
                    metric_monitor.update_values(sum_v, sum_reward, loss, mean_tde, st)

                episode.set_postfix(rewards=sum_reward, loss=loss, nb_action=st, V=sum_v / st, tde=mean_tde)

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
            metrics_to_pdf(vs, rewards, losses, td_errors, nb_action, nb_conv_action, path_plot, "training")

    def evaluate(self, eval_path, result_path='.', plot_metric=False):

        self.img_path = os.path.join(eval_path, "img")
        self.label_path = os.path.join(eval_path, "bboxes")

        self.img_list = sorted(os.listdir(self.img_path))
        self.label_list = sorted(os.listdir(self.label_path))

        # for plotting
        rewards = []
        vs = []
        nb_action = []
        nb_conv_action = []

        # --------------------------------------------------------------------------------------------------------------
        # EVALUATION STEPS
        # --------------------------------------------------------------------------------------------------------------
        with tqdm(range(len(self.img_list)), unit="episode") as episode:
            for i in episode:
                img_filename = self.img_list[i]
                img = os.path.join(self.img_path, img_filename)
                bb = os.path.join(self.label_path, img_filename.split('.')[0] + '.txt')

                first_state = self.env.reload_env(img, bb)
                sum_reward, sum_v = self.agent.exploit_one_episode(first_state)

                rewards.append(sum_reward)
                vs.append(sum_v)
                st = self.env.nb_actions_taken
                nb_action.append(st)
                nb_conv_action.append(self.env.conventional_policy_nb_step)

                episode.set_postfix(rewards=sum_reward, nb_action=st, V=sum_v / st)


        if plot_metric:
            path = os.path.join(result_path, "rts_runs/evaluation")
            path_plot = os.path.join(path, "plot/")
            try:
                os.mkdir(path)
                os.mkdir(path_plot)
            except OSError as error:
                print(error)

            metrics_eval_to_pdf(vs, rewards, nb_action, nb_conv_action, path_plot, "evaluation")
