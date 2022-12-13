"""
The goal of this program is to allow user to evaluate 3 different RL algorithm on the dummy environment.
"""
import argparse

from components.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This program allow user to train RTS on a object detection dataset')
    parser.add_argument('-tr', '--train_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-ts', '--eval_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-o', '--results_path',
                        default=".",
                        help='the path where results will be saved')
    parser.add_argument('-e', '--episodes', help=' number of episodes')

    parser.add_argument("-rt", '--real_time_monitor', default=False, action="store_true")
    parser.add_argument("-plt", '--plot_metric', default=False, action="store_true")
    parser.add_argument("-tl", '--transfer_learning', default=None)

    args = parser.parse_args()

    trainer = Trainer()

    print("[INFO] Training started")
    trainer.train(int(args.episodes),
                  args.train_path,
                  args.results_path,
                  real_time_monitor=args.real_time_monitor,
                  plot_metric=args.plot_metric,
                  transfer_learning=args.transfer_learning)

    print("[INFO Evaluation started")
    trainer.evaluate(args.eval_path,
                     args.results_path,
                     plot_metric=args.plot_metric)
