"""
This file contain the implementation of the real environment.
"""
import math
import os
import random
import re
import cv2
import imageio
import numpy as np
import gc
from matplotlib import pyplot as plt

MODEL_RES = 64
TASK_MODEL_RES = 100
ZOOM_DEPTH = 4


class PriorityQueue(object):
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def append(self, node, rank):
        self.queue.append([rank, node])

    # for popping an element based on Priority
    def pop(self):
        try:
            max_val_i = 0

            for i in range(len(self.queue)):
                if self.queue[i][0] >= self.queue[max_val_i][0]:
                    max_val_i = i

            return self.queue.pop(max_val_i)[1]
        except IndexError:
            print("error")
            exit()


class Node:
    def __init__(self, img, pos, parent, number):
        x, y = pos
        self.number = number
        self.parent = parent
        self.img = img
        self.resized_img = cv2.resize(img, (MODEL_RES, MODEL_RES)) / 255.
        self.x = x
        self.y = y
        self.proba = None
        self.V = None
        self.nb_children = 0

    def get_state(self):
        return np.array(self.resized_img.squeeze())

    def image_limit_attain(self):
        return self.img.shape[0] / 2 <= TASK_MODEL_RES

    def get_child(self, action, number):

        self.nb_children += 1

        h = int(self.img.shape[0] / 2)
        w = int(self.img.shape[1] / 2)

        if action == 0:
            i = 0
            j = 0
        elif action == 1:
            i = 1
            j = 0
        elif action == 2:
            i = 0
            j = 1
        else:
            i = 1
            j = 1

        x_ = self.x + (i * w)
        y_ = self.y + (j * h)

        return Node(self.img[h * j:h + h * j, w * i: w + w * i], (x_, y_), self.number, number)


def check_cuda():
    """
    check if opencv can use cuda
    :return: return True if opencv can detect cuda. False otherwise.
    """
    cv_info = [re.sub('\s+', ' ', ci.strip()) for ci in cv2.getBuildInformation().strip().split('\n')
               if len(ci) > 0 and re.search(r'(nvidia*:?)|(cuda*:)|(cudnn*:)', ci.lower()) is not None]
    return len(cv_info) > 0


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self):
        self.min_res = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.nb_max_conv_action = None
        self.full_img = None
        self.dim = None
        self.pq = None
        self.sub_images_queue = None
        self.current_node = None
        self.Queue = None
        self.conventional_policy_nb_step = None
        self.base_img = None
        self.nb_actions_taken = 0
        self.action_space = 4
        self.cv_cuda = check_cuda()

    def reload_env(self, img, bb):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        self.objects_coordinates = []
        self.prepare_img(img)
        self.prepare_coordinates(bb)

        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.min_zoom_action = 0
        self.current_node = Node(self.full_img, (0, 0), -1, self.nb_actions_taken)

        return self.current_node.get_state()

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        H, W, channels = self.full_img.shape
        # check which dimention is the bigger
        max_ = np.max([W, H])
        # check that the image is divisble by 2
        if max_ % 2:
            max_ += 1

        self.full_img = cv2.copyMakeBorder(self.full_img, 0, max_ - H, 0,
                                           max_ - W, cv2.BORDER_CONSTANT, None, value=0)

        self.dim = max_
        self.min_res = self.dim
        self.nb_zoom_max = 0
        while True:
            if self.min_res / 2 <= TASK_MODEL_RES:
                break
            else:
                self.nb_zoom_max += 1
                self.min_res /= 2

        self.nb_max_conv_action = (self.dim / self.min_res) ** 2

    def prepare_coordinates(self, bb):
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        for line in lines:
            if line is not None:
                values = line.split()
                self.objects_coordinates.append((float(values[0]), float(values[1])))

    def follow_policy(self, probs, V):
        A = np.random.choice(self.action_space, p=probs)
        p = probs[A]
        probs[A] = 0.
        giveaway = p / (np.count_nonzero(probs) + 0.00000001)
        probs[probs != 0.] += giveaway
        self.current_node.proba = probs
        self.current_node.V = V
        return A

    def exploit(self, probs, V):
        A = np.argmax(probs)
        probs[A] = 0.
        self.current_node.proba = probs
        self.current_node.V = V
        return A

    def calc_conventional_policy_step(self, x, y):

        nb_line = self.dim / self.min_res
        nb_col = int(x / self.min_res)
        last = int(y / self.min_res)
        self.conventional_policy_nb_step = nb_line * nb_col + last + 1

    def sub_img_contain_object(self, x, y, window):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """

        for coordinate in self.objects_coordinates:
            o_x, o_y = coordinate

            if x <= o_x <= x + window and y <= o_y <= y + window:
                self.calc_conventional_policy_step(o_x, o_y)
                return True
        return False

    def take_action(self, action):

        reward = 0.
        self.nb_actions_taken += 1
        is_terminal = False

        parent_n = self.current_node.parent
        current_n = self.current_node.number

        child = self.current_node.get_child(action, self.nb_actions_taken)

        node_info = (parent_n, current_n, self.nb_actions_taken)

        if self.current_node.nb_children < 4:
            self.pq.append(self.current_node, self.current_node.V)

        if not child.image_limit_attain():
            self.pq.append(child, self.current_node.V)

        if self.current_node.nb_children >= 4:
            del self.current_node

        if self.pq.isEmpty():
            is_terminal = True

        if child.image_limit_attain():

            self.min_zoom_action += 1
            if self.sub_img_contain_object(child.x, child.y, child.img.shape[0]):
                reward = 10
                is_terminal = True

        if not self.pq.isEmpty():
            self.current_node = self.pq.pop()
        else:
            self.current_node = child
        S_prime = self.current_node.get_state()

        return S_prime, reward, is_terminal, node_info, (self.current_node.proba, self.current_node.V)
