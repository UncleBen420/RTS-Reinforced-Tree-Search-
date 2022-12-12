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
ZOOM_DEPTH = 3


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


class Tree:
    def __init__(self, img, pos, parent, number):
        x, y, z = pos
        self.number = number
        self.childs = []
        self.parent = parent
        self.visited = False
        self.img = img
        self.resized_img = cv2.resize(img, (MODEL_RES, MODEL_RES)) / 255.
        self.x = x
        self.y = y
        self.z = z
        self.proba = None
        self.V = None
        self.nb_childs = 0

    def get_state(self):
        return np.array(self.resized_img.squeeze())

    def get_child(self, action, number):

        self.nb_childs += 1

        sub_z = self.z - 1
        sub_x = self.x << 1
        sub_y = self.y << 1

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

        x_ = sub_x + i
        y_ = sub_y + j

        return Tree(self.img[h * j:h + h * j, w * i: w + w * i], (x_, y_, sub_z), self.number, number)


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
        self.min_zoom = None
        self.max_zoom = None
        self.full_img = None
        self.H = None
        self.W = None
        self.pq = None
        self.sub_images_queue = None
        self.current_node = None
        self.Queue = None
        self.conventional_policy_nb_step = None
        self.root = None
        self.base_img = None
        self.nb_actions_taken = 0
        self.zoom_padding = 2
        self.action_space = 4
        self.cv_cuda = check_cuda()

    def reload_env(self, img, bb):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        self.bboxes = []
        self.prepare_img(img)
        self.prepare_bounding_boxes(bb)

        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.current_node = Tree(self.full_img, (0, 0, self.max_zoom), -1, self.nb_actions_taken)

        return self.current_node.get_state()

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        self.H, self.W, self.channels = self.full_img.shape
        # check which dimention is the bigger
        max_ = np.max([self.W, self.H])
        # check that the image is divisble by 2
        self.max_zoom = int(math.log(max_, 2))
        if 2 ** self.max_zoom < max_:
            self.max_zoom += 1
            max_ = 2 ** self.max_zoom

        self.full_img = cv2.copyMakeBorder(self.full_img, 0, max_ - self.H, 0,
                                           max_ - self.W, cv2.BORDER_CONSTANT, None, value=0)

        self.W = max_
        self.H = max_

        self.max_zoom = int(math.log(self.W, 2))
        self.min_zoom = self.max_zoom - ZOOM_DEPTH

    def prepare_bounding_boxes(self, bb):
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        for line in lines:
            if line is not None:
                x, y, w, h = line.split()
                self.bboxes.append((float(x), float(y), float(w), float(h)))

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

    def sub_img_contain_object(self, x, y, z):
        """
        This method allow the user to know if the current subgrid contain charlie or not
        :return: true if the sub grid contains charlie.
        """

        for bbox in self.bboxes:

            bb_x, bb_y, bb_w, bb_h = bbox
            window = self.zoom_padding << (z - 1)
            bb_w /= 2
            bb_h /= 2

            if ((x * window <= bb_x < x * window + window - bb_w or
                 x * window <= bb_x + bb_w < x * window + window)
                and
                (y * window <= bb_y < y * window + window - bb_h or
                 y * window <= bb_y + bb_h < y * window + window)):
                return True
        return False

    def take_action(self, action):

        reward = -1.
        self.nb_actions_taken += 1
        is_terminal = False

        parent_n = self.current_node.parent
        current_n = self.current_node.number

        child = self.current_node.get_child(action, self.nb_actions_taken)

        node_info = (parent_n, current_n, self.nb_actions_taken)

        if self.current_node.nb_childs < 4:
            self.pq.append(self.current_node, self.current_node.V)

        if not self.current_node.z <= self.min_zoom:
            self.pq.append(child, self.current_node.V)

        if self.pq.isEmpty():
            is_terminal = True

        x = child.x
        y = child.y
        z = child.z

        if z < self.min_zoom and self.sub_img_contain_object(x, y, z):
            reward = 100.
            is_terminal = True

        if not self.pq.isEmpty():
            self.current_node = self.pq.pop()

        S_prime = self.current_node.get_state()

        return S_prime, reward, is_terminal, node_info, (self.current_node.proba, self.current_node.V)
