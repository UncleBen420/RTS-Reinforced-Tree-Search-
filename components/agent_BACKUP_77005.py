import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR


class PolicyNet(nn.Module):
    def __init__(self, img_res=64):
        super(PolicyNet, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.action_space = np.arange(4)

        self.img_res = img_res
        self.sub_img_res = int(self.img_res / 2)

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

        self.head = torch.nn.Sequential(

<<<<<<< HEAD
        self.value_head = torch.nn.Sequential(
            torch.nn.Linear(n_hidden_nodes >> 3, 2)
=======
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(64),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 4)
>>>>>>> benchmarking
        )

        self.backbone.to(self.device)
        self.head.to(self.device)

        self.head.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def prepare_data(self, state):
        img = state.permute(0, 3, 1, 2)
        return img

    def forward(self, state):
        x = self.backbone(state)
        return self.head(x)


class PolicyGradient:

<<<<<<< HEAD
    def __init__(self, environment, learning_rate=0.005, gamma=0.7,
                 entropy_coef=0.3, beta_coef=0.3,
                 lr_gamma=0.7, batch_size=64, pa_dataset_size=2048, pa_batch_size=2, img_res=64):
=======
    def __init__(self, environment, learning_rate=0.0001, gamma=0.5,
                 lr_gamma=0.7, pa_dataset_size=5000, pa_batch_size=300, img_res=64):
>>>>>>> benchmarking

        self.V_pa_batch = None
        self.gamma = gamma
        self.environment = environment
        self.policy = PolicyNet(img_res=img_res)
        self.action_space = 4
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

    def update_policy(self):

        if len(self.A_pa_batch) < self.pa_batch_size:
            return 0.

        shuffle_index = torch.randperm(len(self.A_pa_batch))
        self.A_pa_batch = self.A_pa_batch[shuffle_index]
        self.G_pa_batch = self.G_pa_batch[shuffle_index]
        self.S_pa_batch = self.S_pa_batch[shuffle_index]

<<<<<<< HEAD
        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
        return loss.item()

    def loss_reinforce(self, rewards, selected_log_probs):
        policy_loss = - (rewards.unsqueeze(1) * selected_log_probs).mean()
        loss = policy_loss

        return loss

    def update_policy(self, batch):

        sum_loss = 0.
        counter = 0.

        S, A, G, TD, A2 = batch

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs, V = self.policy(S)
        action_probs = torch.nn.functional.softmax(action_probs, dim=1)
        V = torch.nn.functional.softmax(V, dim=1)

        log_probs = torch.log(action_probs)
        log_probs[torch.isinf(log_probs)] = 0
        selected_log_probs = torch.gather(log_probs, 1, A.unsqueeze(1))
        loss = self.loss_reinforce(G, selected_log_probs)

        log_probs = torch.log(V)
        log_probs[torch.isinf(log_probs)] = 0
        selected_log_probs = torch.gather(log_probs, 1, A2.unsqueeze(1))
        loss += self.loss_reinforce(G, selected_log_probs)
        # sum_loss += self.a2c(TD, G, action_probs, log_probs, selected_log_probs, V)

        loss.backward()

        # torch.nn.utils.clip_grad_value_(self.policy.parameters(), 100.)
        self.optimizer.step()
=======
        S = self.S_pa_batch[:self.pa_batch_size]
        A = self.A_pa_batch[:self.pa_batch_size]
        G = self.G_pa_batch[:self.pa_batch_size]

        # Calculate loss
        self.optimizer.zero_grad()
        action_probs = self.policy(S)

        selected = torch.gather(action_probs, 1, A.unsqueeze(1))
        loss = torch.nn.functional.mse_loss(selected.squeeze(), G)
        loss.backward()
>>>>>>> benchmarking

        self.optimizer.step()
        self.scheduler.step()

<<<<<<< HEAD
        return loss.item() / counter
=======
        return loss.item()
>>>>>>> benchmarking

    def calculate_advantage_tree(self, rewards):
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

        # ------------------------------------------------------------------------------------------------------
        # EPISODE PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = []
        R_batch = []
        A_batch = []
        V_batch = []

        # ------------------------------------------------------------------------------------------------------
        # EPISODE REALISATION
        # ------------------------------------------------------------------------------------------------------
        counter = 0
        sum_reward = 0
<<<<<<< HEAD
=======
        existing_proba = None
>>>>>>> benchmarking

        while True:

            counter += 1
            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

<<<<<<< HEAD
            with torch.no_grad():
                action_probs, V = self.policy(S)
                action_probs = torch.nn.functional.softmax(action_probs, dim=-1)
                action_probs = action_probs.detach().cpu().numpy()[0]
                V = torch.nn.functional.softmax(V, dim=-1)
                V = V.detach().cpu().numpy()[0]
                V = np.random.choice(2, p=V)

            # the environment choose the action because the agent have to respect some rules
            A = self.environment.follow_policy(action_probs)

            sum_v += V

            S_prime, R, is_terminal, node_info = self.environment.take_action(A, V)
=======
            if existing_proba is None:
                with torch.no_grad():
                    action_probs = self.policy(S)
                    action_probs = action_probs.detach().cpu().numpy()[0]

            else:
                action_probs = existing_proba

            A, Q = self.environment.follow_policy(action_probs)

            S_prime, R, is_terminal, node_info, existing_proba = self.environment.take_action(A, Q)
>>>>>>> benchmarking
            parent, current, child = node_info

            S_batch.append(S)
            A_batch.append(A)
<<<<<<< HEAD
            V_batch.append(V)
            R_batch.append((parent, current, child, R, V))
=======
            R_batch.append((parent, current, child, R))
>>>>>>> benchmarking
            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        # ------------------------------------------------------------------------------------------------------
        # CUMULATED REWARD CALCULATION AND TD ERROR
        # ------------------------------------------------------------------------------------------------------
        G_batch = self.calculate_advantage_tree(R_batch)

        # ------------------------------------------------------------------------------------------------------
        # BATCH PREPARATION
        # ------------------------------------------------------------------------------------------------------
        S_batch = torch.concat(S_batch).to(self.policy.device)
        A_batch = torch.LongTensor(A_batch).to(self.policy.device)
        G_batch = torch.FloatTensor(G_batch).to(self.policy.device)
<<<<<<< HEAD
        V_batch = torch.LongTensor(V_batch).to(self.policy.device)
        TDE_batch = torch.FloatTensor(TDE_batch).to(self.policy.device)

        # TD error is scaled to ensure no exploding gradient
        # also it stabilise the learning : https://arxiv.org/pdf/2105.05347.pdf
        self.min_r = min(torch.min(G_batch), self.min_r)
        self.max_r = max(torch.max(G_batch), self.max_r)
        G_batch = self.minmax_scaling(G_batch)
=======
>>>>>>> benchmarking

        # ------------------------------------------------------------------------------------------------------
        # PAST ACTION DATASET PREPARATION
        # ------------------------------------------------------------------------------------------------------
<<<<<<< HEAD
        # Append the past action batch to the current batch if possible

        if self.A_pa_batch is not None and len(self.A_pa_batch) > self.pa_batch_size:
            batch = (torch.cat((self.S_pa_batch[0:self.pa_batch_size], S_batch), 0),
                     torch.cat((self.A_pa_batch[0:self.pa_batch_size], A_batch), 0),
                     torch.cat((self.G_pa_batch[0:self.pa_batch_size], G_batch), 0),
                     torch.cat((self.TDE_pa_batch[0:self.pa_batch_size], TDE_batch), 0),
                     torch.cat((self.V_pa_batch[0:self.pa_batch_size], V_batch), 0))
        else:
            batch = (S_batch, A_batch, G_batch, TDE_batch, V_batch)
=======
>>>>>>> benchmarking

        # Add some experiences to the buffer with respect of TD error
        nb_new_memories = int(counter / 4)

        if nb_new_memories == 0:
            loss = self.update_policy()
            return loss, sum_reward

        weights = G_batch + abs(G_batch.min().item()) + 1
        weights /= weights.sum()
        idx = torch.multinomial(weights, nb_new_memories)


<<<<<<< HEAD
        idx = torch.randperm(len(A_batch))[:nb_new_memories]
        #idx = torch.multinomial(1 - TDE_batch, nb_new_memories, replacement=True)
=======
>>>>>>> benchmarking
        if self.A_pa_batch is None:
            self.A_pa_batch = A_batch[idx]
            self.S_pa_batch = S_batch[idx]
            self.G_pa_batch = G_batch[idx]
<<<<<<< HEAD
            self.V_pa_batch = V_batch[idx]
            self.TDE_pa_batch = TDE_batch[idx]
=======
>>>>>>> benchmarking
        else:
            self.A_pa_batch = torch.cat((self.A_pa_batch, A_batch[idx]), 0)
            self.S_pa_batch = torch.cat((self.S_pa_batch, S_batch[idx]), 0)
            self.G_pa_batch = torch.cat((self.G_pa_batch, G_batch[idx]), 0)
<<<<<<< HEAD
            self.V_pa_batch = torch.cat((self.V_pa_batch, V_batch[idx]), 0)
            self.TDE_pa_batch = torch.cat((self.TDE_pa_batch, TDE_batch[idx]), 0)
=======
>>>>>>> benchmarking

        # clip the buffer if it's to big
        if len(self.A_pa_batch) > self.pa_dataset_size:
            # shuffling the batch
<<<<<<< HEAD
            shuffle_index = torch.randperm(len(self.A_pa_batch))
            self.A_pa_batch = self.A_pa_batch[shuffle_index]
            self.G_pa_batch = self.G_pa_batch[shuffle_index]
            self.S_pa_batch = self.S_pa_batch[shuffle_index]
            self.V_pa_batch = self.V_pa_batch[shuffle_index]
            self.TDE_pa_batch = self.TDE_pa_batch[shuffle_index]
=======
>>>>>>> benchmarking

            # dataset clipping
            surplus = len(self.A_pa_batch) - self.pa_dataset_size
            _, self.A_pa_batch = torch.split(self.A_pa_batch, [surplus, self.pa_dataset_size])
            _, self.G_pa_batch = torch.split(self.G_pa_batch, [surplus, self.pa_dataset_size])
            _, self.S_pa_batch = torch.split(self.S_pa_batch, [surplus, self.pa_dataset_size])
<<<<<<< HEAD
            _, self.V_pa_batch = torch.split(self.V_pa_batch, [surplus, self.pa_dataset_size])
            _, self.TDE_pa_batch = torch.split(self.TDE_pa_batch, [surplus, self.pa_dataset_size])
=======
>>>>>>> benchmarking

        # ------------------------------------------------------------------------------------------------------
        # MODEL OPTIMISATION
        # ------------------------------------------------------------------------------------------------------
        loss = self.update_policy()

        return loss, sum_reward

    def exploit_one_episode(self, S):
        sum_reward = 0

<<<<<<< HEAD
=======
        existing_proba = None

>>>>>>> benchmarking
        while True:

            # State preprocess
            S = torch.from_numpy(S).float()
            S = S.unsqueeze(0).to(self.policy.device)
            S = self.policy.prepare_data(S)

<<<<<<< HEAD
            with torch.no_grad():
                probs, V = self.policy(S)
                probs = torch.nn.functional.softmax(probs, dim=-1)
                probs = probs.detach().cpu().numpy()[0]
                V = torch.nn.functional.softmax(V, dim=-1)
                V = V.detach().cpu().numpy()[0]
                V = np.argmax(V)

            A = self.environment.exploit(probs)
            S_prime, R, is_terminal, _, = self.environment.take_action(A, V)
=======
            if existing_proba is None:
                with torch.no_grad():
                    action_probs = self.policy(S)
                    action_probs = action_probs.detach().cpu().numpy()[0]

            else:
                action_probs = existing_proba

            A, Q = self.environment.follow_policy(action_probs)
            S_prime, R, is_terminal, _, existing_proba = self.environment.take_action(A, Q)
>>>>>>> benchmarking

            sum_reward += R

            S = S_prime

            if is_terminal:
                break

        return sum_reward
