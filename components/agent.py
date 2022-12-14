import time
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm


class PolicyNet(nn.Module):
    def __init__(self, img_res=64, n_hidden_nodes=64, n_kernels=32):
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

        self.middle.apply(self.init_weights)
        self.head.apply(self.init_weights)
        self.value_head.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

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

    def follow_policy(self, probs):
        return np.random.choice(self.action_space, p=probs.detach().cpu().numpy()[0])


class PolicyGradient:

    def __init__(self, environment, learning_rate=0.0001,
                 episodes=100, gamma=0.7,
                 entropy_coef=0.1, beta_coef=0.01,
                 lr_gamma=0.9, batch_size=64, pa_dataset_size=256, pa_batch_size=64, img_res=64):

        self.gamma = gamma
        self.environment = environment
        self.episodes = episodes
        self.beta_coef = beta_coef
        self.entropy_coef = entropy_coef
        self.min_r = 0
        self.max_r = 1
        self.policy = PolicyNet(img_res=img_res)
        self.action_space = 4
        self.batch_size = batch_size
        self.pa_dataset_size = pa_dataset_size
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=lr_gamma)
        self.pa_batch_size = pa_batch_size

        # Past Actions Buffer
        self.S_pa_batch = None
        self.A_pa_batch = None
        self.TDE_pa_batch = None
        self.G_pa_batch = None

    def save(self, file):
        torch.save(self.policy.state_dict(), file)

    def load(self, weights):
        self.policy.load_state_dict(torch.load(weights))

    def model_summary(self):
        print("RUNNING ON {0}".format(self.policy.device))
        print(self.policy)
        print("TOTAL PARAMS: {0}".format(sum(p.numel() for p in self.policy.parameters())))

    def minmax_scaling(self, x):
        return (x - self.min_r) / (self.max_r - self.min_r)

    def a2c(self, advantages, rewards, action_probs, log_probs, selected_log_probs, values):

        entropy_loss = self.entropy_coef * (action_probs * log_probs).sum(1).mean()
        value_loss = self.beta_coef * torch.nn.functional.mse_loss(values.squeeze(), rewards)
        policy_loss = - (advantages.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss + entropy_loss + value_loss
        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
        return loss.item()

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TD = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, V = self.policy(S)
        action_probs = torch.nn.functional.softmax(action_probs, dim=1)

        log_probs = torch.log(action_probs)
        log_probs[torch.isinf(log_probs)] = 0

        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))

        sum_loss += self.a2c(TD, G, action_probs, log_probs, selected_log_probs, V)

        counter += 1
        self.scheduler.step()

        return sum_loss / counter

    def calculate_advantage_tree(self, rewards):
        rewards = np.array(rewards)

        # calculate the discount rewards
        G = rewards[:, 3].astype(float)
        V = rewards[:, 4].astype(float)
        node_info = rewards[:, 0:3].astype(int)

        TDE = V.copy()

        for ni in node_info[::-1]:
            parent, current, child = ni
            parent_index = np.all(node_info[:, [1, 2]] == [parent, current], axis=1)
            current_index = np.all(node_info[:, [1, 2]] == [current, child], axis=1)

            if parent != -1:
                G[parent_index] += self.gamma * G[current_index]
                TDE[parent_index] += self.gamma * V[current_index]

        # calculate the TD error as A = Q(S,A) - V(S) => A + V(S') - V(S)
        return G.tolist(), (G + (TDE - 2 * V)).tolist()

    def fit_one_episode(self, S):

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
        sum_v = 0
        sum_reward = 0
        existing_proba = None
        existing_v = None
        while True:

            counter += 1
            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            if existing_proba is None:
                with torch.no_grad():
                    action_probs, V = self.policy(S)
                    action_probs = torch.nn.functional.softmax(action_probs, dim=-1)
                    action_probs = action_probs.detach().cpu().numpy()[0]
                    V = V.item()
            else:
                action_probs = existing_proba
                V = existing_v

            action_probs /= action_probs.sum()  # adjust probabilities
            A = self.environment.follow_policy(action_probs, V)

            sum_v += V

            S_prime, R, is_terminal, node_info, existing_pred = self.environment.take_action(A)
            existing_proba, existing_v = existing_pred
            parent, current, child = node_info

            S_batch.append(S)
            A_batch.append(A)
            R_batch.append((parent, current, child, R, V))
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        # ------------------------------------------------------------------------------------------------------
        # CUMULATED REWARD CALCULATION AND TD ERROR
        # ------------------------------------------------------------------------------------------------------
        G_batch, TDE_batch = self.calculate_advantage_tree(R_batch)

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
        TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)

        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = min(torch.min(TDE_batch), self.min_r)
        self.max_r = max(torch.max(TDE_batch), self.max_r)
        TDE_batch = self.minmax_scaling(TDE_batch)

        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
        # Append the past action batch to the current batch if possible

        if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
            batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                     torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                     torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0),
                     torch.cat((self.TDE_pa_batch[0:self.pa_batch_size], TDE_batch), 0))
        else:
            batch = (S_batch, A_batch, G_batch, TDE_batch)

        # Add some experiences to the buffer with respect of 1 - TD error
        nb_new_memories = min(5, counter)
        idx = torch.multinomial(1 - TDE_batch, nb_new_memories, replacement=True)
        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
            self.TDE_pa_batch = TDE_batch[idx]
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)
            self.TDE_pa_batch = torch.cat((self.TDE_pa_batch, TDE_batch[idx]), 0)

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch
            shuffle_index = torch.randperm(len(self.A_pa_batch))
            self.A_pa_batch = self.A_pa_batch[shuffle_index]
            self.G_pa_batch = self.G_pa_batch[shuffle_index]
            self.S_pa_batch = self.S_pa_batch[shuffle_index]
            self.TDE_pa_batch = self.TDE_pa_batch[shuffle_index]

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])
            _, self.TDE_pa_batch = torch.split(self.TDE_pa_batch, [surplus, self.pa_dataset_size])

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss = self.update_policy(batch)

        return loss, sum_reward, sum_v, torch.sum(TDE_batch).item()

    def exploit_one_episode(self, S):
        sum_reward = 0
        sum_V = 0

        existing_proba = None
        existing_v = None
        start_time = time.time()
        while True:
            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

            if existing_proba is None:
                with torch.no_grad():
                    probs, V = self.policy(S)
                    probs = torch.nn.functional.softmax(probs, dim=-1)
                    probs = probs.detach().cpu().numpy()[0]
                    V = V.item()
            else:
                probs = existing_proba
                V = existing_v

            # no need to explore, so we select the most probable action
            probs /= probs.sum()
            A = self.environment.exploit(probs, V)
            S_prime, R, is_terminal, _, existing_pred = self.environment.take_action(A)
            existing_proba, existing_v = existing_pred

            S = S_prime
            sum_reward += R
            sum_V += V
            if is_terminal:
                break

        return sum_reward, sum_V
