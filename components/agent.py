import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


class QNet(nn.Module):
    """
    Implementation of a QNET for the RTS algorithm.
    """
    def __init__(self, img_res=64):
        super(QNet, self).__init__()

        # if gpu available, the network is placed on it.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.action_space = np.arange(4)
        self.img_res = img_res

        # feature extractor
        self.backbone = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(7, 7), stride=( 3, 3)),
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

        # mlp
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

        self.head.apply(self.init_weights)

    def init_weights(self, m):
        """
        Init weight in linear layers.
        """
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        """
        Prepare the state receive from the environment.
        @param state: the state receive by the environment.
        """
        img = state.permute(0, 3, 1, 2)
        return img

    def forward(self, state):
        """
        Reimplementation of the forward function.
        @param state: prepared state.
        """
        x = self.backbone(state)
        return self.head(x)


class PolicyGradient:
    """
    The Deep RL algorithm used in RTS. In fact, it is a modified version of a Policy Gradient Descent.
    """

    def __init__(self, environment, learning_rate=0.0001, gamma=0.5,
                 lr_gamma=0.7, pa_dataset_size=5000, pa_batch_size=300, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.policy = QNet(img_res=img_res)
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
        self.pa_batch_size = pa_batch_size

        # Replay Memory Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.TDE_pa_batch = None
        self.G_pa_batch = None

    def save(self, file):
        """
        save the model weight in a .pt file
        @param file: the name given to the .pt file.
        """
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        """
        load weight from a .pt file.
        @param weights: the file containing weights.
        """
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        """
        print a summary of the model including the number of parameters.
        """
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def update_policy(self):
        """
        Upade the model weights using the replay memory.
        @return: the mse loss of the QNet
        """
        if len(self.A_pa_batch) < self.pa_batch_size:
            return 0.

        shuffle_index = torch.randperm(len(self.A_pa_batch))
        self.A_pa_batch = self.A_pa_batch[shuffle_index]
        self.G_pa_batch = self.G_pa_batch[shuffle_index]
        self.S_pa_batch = self.S_pa_batch[shuffle_index]

        S = self.S_pa_batch[:self.pa_batch_size]
        A = self.A_pa_batch[:self.pa_batch_size]
        G = self.G_pa_batch[:self.pa_batch_size]

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        selected = torch.gather(action_probs, 1, A.unsqueeze(1))
        loss = torch.nn.functional.mse_loss(selected.squeeze(), G)
        loss.backward()

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def calculate_advantage_tree(self, rewards):
        """
        Calculate the return (G) in a tree like shape.
        """
        rewards = np.array(rewards)

        # calculate the discount rewards
        G = rewards[:, 3].astype(float)
        node_info = rewards[:, 0:3].astype(int)

        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                G[parent_index] += self.gamma * G[current_index]

        return G.tolist()

    def fit_one_episode(self, S):
        """
        Train the model for 1 episode only.
        @param S: initial state of the environment.
        """

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_reward = 0
        existing_proba = None

        while True:

            counter += 1
            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            # the current node already possess prediction there is no need to predict a second time
            if existing_proba is None:
                with torch.no_grad():
                    action_probs = self.policy(S)
                    action_probs = action_probs.detach().cpu().numpy()[0]
            else:
                action_probs = existing_proba

            # use the e-greedy policy
            A, Q = self.environment.follow_policy(action_probs)

            # take an action on the environment.
            S_prime, R, is_terminal, node_info, existing_proba = self.environment.take_action(A, Q)
            parent, current, child = node_info

            # add step to the episode buffer
            S_batch.append(S)
            A_batch.append(A)
            R_batch.append((parent, current, child, R))
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        # ------------------------------------------------------------------------------------------------------
        # CUMULATED REWARD CALCULATION
        # ------------------------------------------------------------------------------------------------------
        G_batch = self.calculate_advantage_tree(R_batch)

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(G_batch).to(self.policy.device)

        # ------------------------------------------------------------------------------------------------------
        # REPLAY MEMORY PREPARATION
        # ------------------------------------------------------------------------------------------------------

        nb_new_memories = int(counter / 4)

        # if there is no step in the episode (this should never be the case),
        # there is no need to add it to the replay memory
        if nb_new_memories == 0:
            loss = self.update_policy()
            return loss, sum_reward

        weights = G_batch + abs(G_batch.min().item()) + 1
        weights /= weights.sum()
        idx = torch.multinomial(weights, nb_new_memories)

        # If there is no element in the replay memory.
        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)

        # clip the buffer if it's too big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss = self.update_policy()

        return loss, sum_reward

    def exploit_one_episode(self, S):
        """
        Exploit on the environment (no more training needed.
        @param S: initial state of the environment.
        """

        sum_reward = 0
        existing_proba = None
        while True:

            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            if existing_proba is None:
                with torch.no_grad():
                    action_probs = self.policy(S)
                    action_probs = action_probs.detach().cpu().numpy()[0]

            else:
                action_probs = existing_proba

            A, Q = self.environment.follow_policy(action_probs)
            S_prime, R, is_terminal, _, existing_proba = self.environment.take_action(A, Q)

            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        return sum_reward
