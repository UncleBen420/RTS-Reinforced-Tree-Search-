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
    parser.add_argument('-ts', '--test_path',
                        help='the path to the data. it must contains a images and a labels folder')
    parser.add_argument('-o', '--saved_model_path',
                        help='the path where the model will be saved')
    parser.add_argument('-e', '--episodes', help=' number of episodes')

    args = parser.parse_args()

    trainer = Trainer()

    trainer.train(int(args.episodes), args.train_path)