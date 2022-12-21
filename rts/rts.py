import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

AGENT_RES = 64
MODEL_RES = 100
ZOOM_DEPTH = 4


class PolicyNet(nn.Module):
    def __init__(self, img_res=64, n_hidden_nodes=64, n_kernels=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(4)

        self.img_res = img_res
        self.sub_img_res = int(self.img_res / 2)

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=3, out_channels=n_kernels >> 3, kernel_size=(1, 9, 9)),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=n_kernels >> 3, out_channels=n_kernels >> 2, kernel_size=(1, 7, 7)),
            torch.nn.ReLU(),
            torch.nn.MaxPool3d((1, 2, 2)),
            torch.nn.Conv3d(in_channels=n_kernels >> 2, out_channels=n_kernels >> 1, kernel_size=(1, 5, 5)),
            torch.nn.Dropout(0.1),
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=n_kernels >> 1, out_channels=n_kernels, kernel_size=(1, 3, 3)),
            torch.nn.Flatten(),
        )

        self.middle = torch.nn.Sequential(
            torch.nn.Linear(n_kernels * 36, n_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes, n_hidden_nodes >> 2),
            torch.nn.ReLU(),
            torch.nn.Linear(n_hidden_nodes >> 2, n_hidden_nodes >> 3),
            torch.nn.ReLU()
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 3, 4)
        )

        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 3, 1)
        )

        self.backbone.to(self.device)
        self.middle.to(self.device)
        self.head.to(self.device)
        self.value_head.to(self.device)

    def prepare_data(self, state):
        img = state.permute(0, 3, 1, 2)
        patches = img.unfold(1, 3, 3).unfold(2, self.sub_img_res, self.sub_img_res).unfold(3, self.sub_img_res,
                                                                                           self.sub_img_res)
        patches = patches.contiguous().view(1, 4, -1, self.sub_img_res, self.sub_img_res)
        patches = patches.permute(0, 2, 1, 3, 4)
        return patches

    def forward(self, state):
        x = self.backbone(state)
        x = self.middle(x)
        return self.head(x), self.value_head(x)


class Agent:

    def __init__(self, img_res=64):

        self.policy = PolicyNet(img_res=img_res)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def predict(self, S):
        # State preprocess
        S = torch.from_numpy(S).float()
        S = S.unsqueeze(0).to(self.policy.device)
        S = self.policy.prepare_data(S)

        # Prediction
        with torch.no_grad():
            probs, V = self.policy(S)
        probs = torch.nn.functional.softmax(probs, dim=-1)

        # to memory
        probs = probs.detach().cpu().numpy()[0]
        V = V.item()

        return probs, V


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
            print("Priority Queue exception")
            exit()


class Node:
    def __init__(self, img, pos):
        x, y = pos
        self.img = img
        self.resized_img = cv2.resize(img, (AGENT_RES, AGENT_RES)) / 255.
        self.x = x
        self.y = y
        self.proba = None
        self.V = None
        self.nb_children = 0

    def get_state(self):
        return np.array(self.resized_img.squeeze())

    def get_model_x(self):
        return cv2.resize(self.img, (MODEL_RES, MODEL_RES)) / 255.

    def image_limit_attain(self):
        return self.img.shape[0] / 2 <= MODEL_RES

    def has_proba(self):
        return self.proba is not None

    def get_next_action(self):
        A = np.argmax(self.proba)
        self.proba[A] = 0.
        return A

    def is_visited(self):
        return not np.count_nonzero(self.proba)

    def get_child(self, action):

        window = int(self.img.shape[0] / 2)

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

        x_ = self.x + (i * window)
        y_ = self.y + (j * window)

        return Node(self.img[window * j:window + window * j, window * i: window + window * i], (x_, y_))


def normalize(values):
    min_val = np.min(values)
    max_val = np.max(values)

    return (values - min_val) / (max_val - min_val + 0.000000001)


class Environment:
    """
    this class implement a problem where the agent must mark the place where he have found boat.
    He must not mark place where there is house.
    """

    def __init__(self, agent, model, collect=False, threshold=0.2):
        self.collection_V = None
        self.collection = None
        self.history = None
        self.sum_V = None
        self.nb_zoom_max = None
        self.min_res = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.nb_max_conv_action = None
        self.full_img = None
        self.dim = None
        self.pq = None
        self.current_node = None
        self.conventional_policy_nb_step = None
        self.nb_actions_taken = 0
        self.agent = agent
        self.model = model
        self.collect = collect
        self.threshold = threshold

    def reload_env(self, img):
        """
        allow th agent to keep the environment configuration and boat placement but reload all the history and
        value to the starting point.
        :return: the current state of the environment.
        """
        self.prepare_img(img)

        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.min_zoom_action = 0
        self.sum_V = 0
        self.history = []
        self.collection = []
        self.collection_V = []
        self.current_node = Node(self.full_img, (0, 0))

    def prepare_img(self, img):
        self.full_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            if self.min_res / 2 <= MODEL_RES:
                break
            else:
                self.nb_zoom_max += 1
                self.min_res /= 2

        self.nb_max_conv_action = (self.dim / self.min_res) ** 2

    def next_step(self):

        self.nb_actions_taken += 1
        is_terminal = False
        model_prediction = None

        if not self.current_node.has_proba():
            S = self.current_node.get_state()
            preds, V = self.agent.predict(S)

            self.current_node.V = V
            self.current_node.proba = preds
            self.sum_V += V

        A = self.current_node.get_next_action()

        child = self.current_node.get_child(A)
        V = self.current_node.V
        pos = (child.x, child.y, child.img.shape[0], V)

        self.history.append(pos)

        if not self.current_node.is_visited() and V > 0:
            self.pq.append(self.current_node, V)

        if not child.image_limit_attain() and V > 0:
            self.pq.append(child, V)

        if self.current_node.is_visited:
            del self.current_node

        if self.pq.isEmpty():
            is_terminal = True
        else:
            self.current_node = self.pq.pop()

        if child.image_limit_attain():
            self.min_zoom_action += 1
            if self.collect:
                self.collection.append((child.get_model_x(), pos))
                self.collection_V.append(V)
            else:
                X = child.get_model_x()
                model_prediction = (self.model(X), pos)

        return is_terminal, model_prediction

    def pred_collection(self):
        Vs = normalize(self.collection_V)

        predictions = []
        idx = np.where(Vs >= self.threshold)[0]
        for i in idx:
            X, pos = self.collection[i]
            self.model(X)
            predictions.append((X, pos))

        return predictions

    def get_history_img(self, history):
        ratio = 255. / self.dim
        hist_img = cv2.resize(self.full_img, (255, 255))
        heat_map = np.zeros((255, 255))

        for step in history:
            x, y, window, V = step
            x *= ratio
            y *= ratio
            window *= ratio
            x = int(x)
            y = int(y)
            window = int(window)

            heat_map[y:window + y, x: window + x] += V

        heat_map /= np.max(heat_map)
        heat_map *= 255

        blur = cv2.GaussianBlur(heat_map.astype(np.uint8), (3, 3), 3)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        return cv2.addWeighted(heatmap_img, 0.4, hist_img, 0.6, 0)


def dynamic_import(module_name, class_name):
    module = __import__(module_name)

    return getattr(module, class_name)


class RTS:

    def __init__(self, model, agent_weights_file="weights_rts.pth", collect=False, threshold=0.1):

        file_dir = os.path.dirname(os.path.abspath(__file__))
        rts_dir = os.path.join(file_dir, 'weights', agent_weights_file)

        self.agent = Agent()
        self.agent.load(rts_dir)
        self.model = model
        self.collect = collect
        self.env = Environment(self.agent, self.model, collect, threshold=threshold)

    def __call__(self, X):
        self.env.reload_env(X)

        predictions = []

        while True:
            is_terminal, prediction = self.env.next_step()
            if prediction is not None:
                predictions.append(prediction)

            if is_terminal:
                break

        if self.collect:
            predictions = self.env.pred_collection()

        return predictions

if __name__ == "__main__":
    img = cv2.imread("test_img2.jpg")

    model = dynamic_import("Dummy_model", "Dummy_model")()

    rts = RTS(model, collect=True)
    preds = rts(img)
    pos = []
    X = []
    for p in preds:
        x, po = p
        pos.append(po)
        X.append(x)

    print(len(pos))
    print(rts.env.sum_V)
    plt.imshow(rts.env.get_history_img(rts.env.history))
    plt.show()

    plt.imshow(rts.env.get_history_img(pos))
    plt.show()

