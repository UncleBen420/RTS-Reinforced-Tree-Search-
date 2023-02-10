import argparse
import os
import cv2
import time
import numpy as np
import torch
from torch import nn

AGENT_RES = 64

class YoloWrapper:
    """
    This class implement a wrapper for YOLO v5n. It allows it to be use wth RTS.
    """
    def __init__(self, model, return_img=False):
        self.model = model
        self.return_img = return_img

    def __call__(self, img):
        """
        Redefinition of the __call__ to call YOLO.
        """
        result = self.model([img]).xyxy[0].cpu().numpy()
        detected = len(result) > 0

        if not self.return_img:
            return result, detected

        for bb in result:
            x_min = int(bb[0])
            y_min = int(bb[1])
            x_max = int(bb[2])
            y_max = int(bb[3])

            img = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), [255, 0, 0], 2)

        return result, detected, img


class QNet(nn.Module):
    def __init__(self, img_res=64):
        super(QNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(4)

        self.img_res = img_res

        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(7, 7), stride=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
        )

        self.backbone.to(self.device)
        self.head.to(self.device)

        self.backbone.to(self.device)
        self.head.to(self.device)

    def prepare_data(self, state):
        img = state.permute(0, 3, 1, 2)
        return img

    def forward(self, state):
        x = self.backbone(state)
        return self.head(x)


class Agent:

    def __init__(self, img_res=64):
        self.policy = QNet(img_res=img_res)

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
            probs = self.policy(S)

        # to memory
        probs = probs.detach().cpu().numpy()[0]

        return probs


class PriorityQueue(object):
    """
    Class implementing a priority queue
    """
    def __init__(self):
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
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
            print("Priority Queue exception")
            exit()


class Node:
    """
    Class implementing a node of the environment. Each node symbolize an action (a zoom in the image).
    """

    def __init__(self, img, pos, model_res=200):
        x, y = pos
        self.img = img
        self.resized_img = cv2.resize(img, (AGENT_RES, AGENT_RES)) / 255.
        self.x = x
        self.y = y
        self.proba = None
        self.Q = None
        self.nb_children = 0
        self.model_res = model_res

    def get_state(self):
        """
        Return an image as a state that can be used by the agent.
        @return: an image.
        """
        return np.array(self.resized_img.squeeze())

    def get_model_x(self):
        """
        Return the sub-image of the node resized in the model resolution.
        @return: an image resized.
        """
        return cv2.resize(self.img, (self.model_res, self.model_res))

    def image_limit_attain(self):
        """
        @return: True if the minimal zoom is attained.
        """
        return self.img.shape[0] / 2 <= self.model_res

    def has_proba(self):
        """
        if the node has already probability assigned to it.
        """
        return self.proba is not None

    def get_next_action(self):
        """
        By respect of the Q predicted by the agent. give the next action.
        """
        A = np.argmax(self.proba)
        Q = self.proba[A]
        self.proba[A] = -1000.
        self.Q = Q
        return A, Q

    def is_visited(self):
        """
        If the node is completely visited.
        """
        return np.all(self.proba <= -1000.)

    def is_not_padding(self):
        """
        if the node is not in the padding area.
        """
        return not np.all(self.resized_img == 0)

    def get_child(self, action):
        """
        compute a new node given the action taken by the agent (in which subpart of the image the agent will zoom).
        @param action: which action to take.
        @return: the new node.
        """

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

        return Node(self.img[window * j:window + window * j, window * i: window + window * i],
                    (x_, y_), model_res=self.model_res)


class Environment:
    """
    this class implement a environment using high resolution image. The agent must find object of
    interest inside the image.
    """

    def __init__(self, agent, model, max_action_allowed=70, model_res=200):
        self.hist_img = None
        self.max_action_allowed = max_action_allowed
        self.history = None
        self.min_zoom_action = None
        self.objects_coordinates = None
        self.full_img = None
        self.dim = None
        self.pq = None
        self.current_node = None
        self.nb_actions_taken = 0
        self.agent = agent
        self.model = model
        self.model_res = model_res

    def reload_env(self, img):
        """
        reload a new environment given an image file.
        @param img: path to the image file.
        """
        self.prepare_img(img)
        self.hist_img = self.full_img.copy()
        
        self.pq = PriorityQueue()
        self.nb_actions_taken = 0
        self.min_zoom_action = 0
        self.history = []
        self.current_node = Node(self.full_img, (0, 0), model_res=self.model_res)

    def prepare_img(self, img):
        """
        prepare the environment with a new image.
        @param img: the path to the image used as the environment.
        """

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

    def next_step(self):
        """
        Find the next zone to explore using the agent predictions.
        """
        self.nb_actions_taken += 1
        is_terminal = False
        model_prediction = None

        if not self.current_node.has_proba():
            S = self.current_node.get_state()
            self.current_node.proba = self.agent.predict(S)

        A, Q = self.current_node.get_next_action()

        child = self.current_node.get_child(A)

        if not self.current_node.is_visited():
            self.pq.append(self.current_node, Q)

        if not child.image_limit_attain() and child.is_not_padding():
            self.pq.append(child, Q)

        if self.current_node.is_visited:
            del self.current_node

        if self.pq.isEmpty():
            is_terminal = True
        else:
            self.current_node = self.pq.pop()

        if self.min_zoom_action >= self.max_action_allowed:
            is_terminal = True

        if child.image_limit_attain() and child.is_not_padding():
            pos = (child.x, child.y, child.img.shape[0], Q)
            self.min_zoom_action += 1
            self.history.append(pos)
            if self.model is not None:
                X = child.get_model_x()
                if self.model.return_img:
                    pred, detect, img = self.model(X)
                    img = cv2.resize(img, (child.img.shape[0], child.img.shape[0]))
                    self.hist_img[child.y: child.y + child.img.shape[0], child.x: child.x + child.img.shape[0]] = img
                else:
                    pred, detect = self.model(X)

                model_prediction = (pos, pred)
            else:
                model_prediction = (pos, '')

        return is_terminal, model_prediction

    def get_history_img(self, history):
        """
        Return an image of the trajectory.
        """
        ratio = 255. / self.dim
        hist_img = cv2.resize(self.full_img, (255, 255))
        heat_map = np.full((255, 255), 0.)

        for step in history:
            x, y, window, Q = step
            x *= ratio
            y *= ratio
            window *= ratio
            x = int(x)
            y = int(y)
            window = int(window)

            heat_map[y:window + y, x: window + x] = 1

        heat_map = (heat_map - np.min(heat_map)) / (np.max(heat_map) - np.min(heat_map))
        heat_map *= 255

        blur = cv2.GaussianBlur(heat_map.astype(np.uint8), (3, 3), 3)
        heatmap_img = cv2.applyColorMap(blur, cv2.COLORMAP_JET)
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(heatmap_img, 0.4, hist_img, 0.6, 0)


class RTS:
    """
    Wrapper of the agent and environment together. Simplify the usage.
    """
    def __init__(self, model=None, agent_weights_file="weights_rts.pt", max_action_allowed=100, model_res=200):

        file_dir = os.path.dirname(os.path.abspath(__file__))
        rts_dir = os.path.join(file_dir, 'weights', agent_weights_file)

        self.agent = Agent()
        self.agent.load(rts_dir)
        self.model = model
        self.env = Environment(self.agent, self.model, max_action_allowed=max_action_allowed, model_res=model_res)

    def __call__(self, X):
        self.env.reload_env(X)

        predictions = {}
        while True:
            is_terminal, prediction = self.env.next_step()
            if prediction is not None:
                pos, model_pred = prediction
                x, y, window, Q = pos
                predictions[(x, y, window)] = model_pred

            if is_terminal:
                break

        return predictions

    def get_trajectory(self):
        """
        Return the trajectory has an image.
        """
        return cv2.cvtColor(self.env.get_history_img(self.env.history), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # TEST WITH YOLO WRAPPER
    # ------------------------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description='This program allow user to use RTS bounded with YOLO v5n to search on a image')
    parser.add_argument('-img', '--image_path',
                        help='the path to the image that will be analyzed')
    parser.add_argument('-max', '--max_actions_allowed', default=100,
                        help='the maximal number of actions allowed to be perform by RTS')
    parser.add_argument('-yolo', '--yolo_weights', default="yolo_weights.pt",
                        help='the name of the yolo weights file (default: yolo_weights.pt)')
    parser.add_argument('-rts', '--rts_weights', default="weights_rts.pt",
                        help='the name of the rts weights file (default: weights_rts.pt)')


    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    file_dir = os.path.dirname(os.path.abspath(__file__))
    yolo_dir = os.path.join(file_dir, 'weights', args.yolo_weights)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=yolo_dir, force_reload=True)
    model = YoloWrapper(model=model, return_img=True)
    rts = RTS(model, agent_weights_file=args.rts_weights, max_action_allowed=int(args.max_actions_allowed),
              model_res=640)
    start = time.time()
    preds = rts(img)
    done = time.time()
    elapsed = done - start
    print("Image computed in {0} secondes".format(elapsed))

    print(preds)

    cv2.imwrite("result.jpg", cv2.cvtColor(rts.env.hist_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("trajectory.jpg", cv2.cvtColor(rts.env.get_history_img(rts.env.history), cv2.COLOR_RGB2BGR))

