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
from matplotlib import pyplot as plt

MODEL_RES = 64
HIST_RES = 200


class PriorityQueue(object):
    """
    Class implementing a priority queue
    """

    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    def isEmpty(self):
        """
        @return: True if the queue is empty
        """
        return len(self.queue) == 0

    # for inserting an element in the queue
    def append(self, node, rank):
        """
        @param node: element that will be added to the queue
        @param rank: rank of the element in the queue
        """
        self.queue.append([rank, node])

    # for popping an element based on Priority
    def pop(self):
        """
        remove the first element of the queue in term of rank and return it.
        @return: the removed element
        """
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
    """
    Class implementing a node of the environment. Each node symbolize an action (a zoom in the image).
    """

    def __init__(self, img, pos, parent, number, res_min=200):
        x, y = pos
        self.number = number
        self.parent = parent
        self.img = img
        self.resized_img = cv2.resize(img, (MODEL_RES, MODEL_RES)) / 255.
        self.x = x
        self.y = y
        self.proba = None
        self.V = 1.
        self.nb_children = 0
        self.res_min = res_min

    def get_state(self):
        """
        Return an image as a state that can be used by the agent.
        @return: an image.
        """
        return np.array(self.resized_img.squeeze())

    def image_limit_attain(self):
        """
        @return: True if the minimal zoom is attained.
        """
        return self.img.shape[0] / 2 <= self.res_min

    def get_child(self, action, number):
        """
        compute a new node given the action taken by the agent (in which subpart of the image the agent will zoom).
        @param action: which action to take.
        @param number: the current count of action taken.
        @return: the new node.
        """

        self.nb_children += 1

        h = int(self.img.shape[0] / 2)
        w = int(self.img.shape[1] / 2)

        if action == 0:  # upper left
            i = 0
            j = 0
        elif action == 1:  # upper right
            i = 1
            j = 0
        elif action == 2:  # down left
            i = 0
            j = 1
        else:  # down right
            i = 1
            j = 1

        x_ = self.x + (i * w)
        y_ = self.y + (j * h)

        return Node(self.img[h * j:h + h * j, w * i: w + w * i], (x_, y_), self.number, number, res_min=self.res_min)


class Environment:
    """
    this class implement a environment using high resolution image. The agent must find object of
    interest inside the image.
    """

    def __init__(self, epsilon, res_min=200):
        self.conventional_policy_nb_step_per_object = None
        self.marked = None
        self.nb_max_conv_action = None
        self.count_per_action = None
        self.history = None
        self.min_res = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.full_img = None
        self.dim = None
        self.pq = None
        self.sub_images_queue = None
        self.current_node = None
        self.Queue = None
        self.conventional_policy_nb_step = None
        self.nb_actions_taken = 0
        self.action_space = 4
        self.e = epsilon
        self.good_hits = 0
        self.res_min = res_min

    def reload_env(self, img, bb):
        """
        reload a new environment given an image file and a bounding box file.
        @param img: path to the image file.
        @param bb: path to the bounding box file.
        """
        self.objects_coordinates = []
        self.history = []
        self.count_per_action = np.zeros(4)
        self.prepare_img(img)
        self.prepare_coordinates(bb)
        self.marked = np.zeros(len(self.objects_coordinates))
        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.conventional_policy_nb_step_per_object = np.zeros(len(self.objects_coordinates))
        self.conventional_policy_nb_step = 0.
        self.min_zoom_action = 0
        self.good_hits = 0
        self.current_node = Node(self.full_img, (0, 0), -1, self.nb_actions_taken, res_min=self.res_min)
        return self.current_node.get_state()

    def prepare_img(self, img):
        """
        prepare the environment with a new image.
        @param img: the path to the image used as the environment.
        """
        self.full_img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
        H, W, channels = self.full_img.shape
        # check which dimention is the bigger
        max_ = np.max([W, H])
        # check that the image is divisble by 2
        if max_ % 2:
            max_ += 1

        self.full_img = cv2.copyMakeBorder(self.full_img, 0, max_ - H, 0,
                                           max_ - W, cv2.BORDER_CONSTANT, None, value=0)

        # compute the number of conventional policy max step
        self.dim = max_
        self.min_res = self.dim
        self.nb_zoom_max = 0
        while self.min_res / 2 > self.res_min:
            self.nb_zoom_max += 1
            self.min_res /= 2

        self.nb_max_conv_action = (self.dim / self.min_res) ** 2

    def prepare_coordinates(self, bb):
        """
        compute the location of objects to detect.
        @param bb: path to the file containing the bounding boxes
        """
        bb_file = open(bb, 'r')
        lines = bb_file.readlines()
        for line in lines:  # for each line in the file, a bounding box is created.
            if line is not None:
                values = line.split()
                self.objects_coordinates.append((float(values[1]), float(values[2]),
                                                 float(values[3]), float(values[4])))

    def follow_policy(self, Qs):
        """
        Determine the next action according to Q and e-greedy policy. This function is not located in the agent side
        because it save prediction in the Node.
        @param Qs: the expected rewards predicted by the agent.
        @return: the action and the expected reward
        """
        p = random.random()
        if p < self.e:
            idx = np.where(Qs > -1000.)[0]
            A = random.choice(idx)
        else:
            A = np.argmax(Qs)
        Q = Qs[A]
        Qs[A] = -1000.  # arbitrary small value, the Q-Net will never predict a value smaller.
        self.current_node.proba = Qs
        self.current_node.V = Q
        return A, Q

    def exploit(self, Qs):
        """
        Determine the next action according to Q and a exploiting policy (argmax). This function is not located in the agent side
        because it save prediction in the Node.
        @param Qs: the expected rewards predicted by the agent.
        @return: the action and the expected reward
        """
        A = np.argmax(Qs)
        Q = Qs[A]
        Qs[A] = -1000.
        self.current_node.proba = Qs
        self.current_node.V = Q
        return A, Q

    def calc_conventional_policy_step(self, x, y):
        """
        Calculate the number of steps needed to attain the same place with the conventional policy.
        @param x: x coordinate of the agent.
        @param y: y coordinate of the agent.
        @return: the number of step.
        """
        nb_line = self.dim / self.min_res
        nb_col = int(x / self.min_res)
        last = int(y / self.min_res)
        return nb_line * nb_col + last + 1

    def sub_img_contain_object(self, x, y, window, mark=False):
        """
        This method allow the user to know if the current subgrid contain a searched object or not
        @param x: the x coordinate of the agent.
        @param y: the y coordinate of the agent.
        @param window: the resolution of the sub image the agent observ.
        @param mark: indicate if the object need to be marked (if all object are marked the episode is over).
        @return: true if the sub grid contains charlie.
        """
        max_iou = 0.
        for i, coordinate in enumerate(self.objects_coordinates):
            o_x, o_y, o_w, o_h = coordinate
            iou = self.intersection_over_union((x, y, window, window), (o_x, o_y, o_w, o_h))
            if iou > 0 and mark:
                if not self.marked[i]:
                    self.marked[i] = self.nb_actions_taken
            if iou > max_iou:
                max_iou = iou
        return max_iou

    def intersection_over_union(self, boxA, boxB):
        """
        This method calculate the intersection over union of 2 bounding boxes.
        @param boxA: a bounding box given by (x, y, w, h).
        @param boxB: a bounding box given by (x, y, w, h).
        @return: the iou.
        """

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2] + boxA[0], boxB[2] + boxB[0])
        yB = min(boxA[3] + boxA[1], boxB[3] + boxB[1])

        # compute the area of intersection rectangle
        interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
        if interArea == 0:
            return 0
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou

    def take_action(self, action, Q):
        """
        Allow the agent to change the environment by taking an action.
        @param action: the action the agent will take.
        @param Q: the expected reward. It is needed to set the rank in the priority queue.
        """

        # setting steps and environment information
        self.count_per_action[action] += 1
        reward = 0.
        self.nb_actions_taken += 1
        is_terminal = False

        # setting node information
        parent_n = self.current_node.parent
        current_n = self.current_node.number
        node_info = (parent_n, current_n, self.nb_actions_taken)
        child = self.current_node.get_child(action, self.nb_actions_taken)

        # append the current step to visualize the trajectory later.
        self.history.append((child.x, child.y, child.img.shape[0], Q))

        # Various check needed
        if self.current_node.nb_children < 4: # if the current node has not been already explored.
            self.pq.append(self.current_node, Q)

        if not child.image_limit_attain() and not np.all(child.resized_img == 0):
            self.pq.append(child, Q)

        # if the current node has been explored it is destroyed
        if self.current_node.nb_children >= 4:
            del self.current_node

        # if the priority queue is empty the episode is over (this should never be the case)
        if self.pq.isEmpty():
            is_terminal = True

        # if the minimal zoom level is attained by the agent.
        if child.image_limit_attain():
            self.min_zoom_action += 1
            if self.sub_img_contain_object(child.x, child.y, child.img.shape[0], mark=True):
                reward = 10.
            else:
                reward = -1.

        # if the zone zoomed in by the agent contain a region of interest.
        if self.sub_img_contain_object(child.x, child.y, child.img.shape[0]):
            self.good_hits += 1

        # if all searched object are marked, the agent has finish the episode.
        if np.all(self.marked > 0):
            # Calculation of the conventional policy metric.
            for i, coordinate in enumerate(self.objects_coordinates):
                o_x, o_y, _, _ = coordinate
                conv_action = self.calc_conventional_policy_step(o_x, o_y)
                self.conventional_policy_nb_step_per_object[i] = conv_action
            if len(self.conventional_policy_nb_step_per_object):
                self.conventional_policy_nb_step = np.max(self.conventional_policy_nb_step_per_object)
            is_terminal = True

        # if the queue is not empty (should always be the case), get the next step by rank in the priority queue.
        if not self.pq.isEmpty():
            self.current_node = self.pq.pop()
        else:
            self.current_node = child
        S_prime = self.current_node.get_state()

        return S_prime, reward, is_terminal, node_info, self.current_node.proba

    def get_gif_trajectory(self, name):
        """
        This function allow the user to create a gif of all the moves the
        agent has made along the episodes
        @param name: the name of the gif file
        """
        frames = []
        H, W, channels = self.full_img.shape
        ratio = HIST_RES / max(H, W)
        hist_img = cv2.resize(self.full_img, (HIST_RES, HIST_RES))
        for step in self.history:
            hist_frame = hist_img.copy()
            x, y, window, Q = step
            x *= ratio
            y *= ratio
            window *= ratio
            x = int(x)
            y = int(y)
            window = int(window)

            if Q > 1.:
                Q = 1.
            elif Q < 0.:
                Q = 0.

            hist_frame[y:window + y, x: window + x] = [int(255 * Q / 100.), 0, int(255 * (1. - Q / 100.))]

            frames.append(hist_frame)

        imageio.mimsave(name, frames, duration=0.2)
