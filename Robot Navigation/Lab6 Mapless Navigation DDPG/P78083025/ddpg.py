import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG():
    def __init__(self, model, learning_rate=[1e-4, 2e-4], reward_decay=0.98, replace_target_iter=300,
                 memory_size=5000, batch_size=64, tau=0.01,
                 epsilon_params=[1.0, 0.5, 0.00001],  # init var / final var / decay
                 criterion=nn.MSELoss()):
        # initialize parameters
        self.lr = learning_rate
        self.gamma = reward_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.tau = tau
        self.criterion = criterion
        self.epsilon_params = epsilon_params
        self.epsilon = self.epsilon_params[0]
        self._build_net(model[0], model[1])
        self.init_memory()

    def _build_net(self, anet, cnet):
        # Policy Network
        self.actor = anet().to(device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr[0])
        # Evaluation Critic Network (new)
        self.critic = cnet().to(device)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr[1])
        # Target Critic Network (old)
        self.critic_target = cnet().to(device)
        self.critic_target.eval()

    def save_load_model(self, op, path):
        anet_path = path + "ddpg_anet.pt"
        cnet_path = path + "ddpg_cnet.pt"
        if op == "save":
            torch.save(self.critic.state_dict(), cnet_path)
            torch.save(self.actor.state_dict(), anet_path)
        elif op == "load":
            self.critic.load_state_dict(torch.load(cnet_path, map_location=device))
            self.critic_target.load_state_dict(torch.load(cnet_path, map_location=device))
            self.actor.load_state_dict(torch.load(anet_path, map_location=device))

    def choose_action(self, s, eval=False):
        # TODO(Lab-03): Apply the noise for exploration.
        ########################################################################################
        s_ts = torch.FloatTensor(np.expand_dims(s, 0)).to(device)
        action = self.actor(s_ts)
        action = action.cpu().detach().numpy()[0]

        if eval == False:  # use epsilon
            action += np.random.normal(0, self.epsilon, action.shape)
        else:  # use final variance
            action += np.random.normal(0, self.epsilon_params[1], action.shape)

        action = np.clip(action, -1, 1)
        ########################################################################################
        return action

    def init_memory(self):
        self.memory_counter = 0
        self.memory = {"s": [], "a": [], "r": [], "sn": [], "end": []}

    def store_transition(self, s, a, r, sn, end):
        if self.memory_counter <= self.memory_size:
            self.memory["s"].append(s)
            self.memory["a"].append(a)
            self.memory["r"].append(r)
            self.memory["sn"].append(sn)
            self.memory["end"].append(end)
        else:
            index = self.memory_counter % self.memory_size
            self.memory["s"][index] = s
            self.memory["a"][index] = a
            self.memory["r"][index] = r
            self.memory["sn"][index] = sn
            self.memory["end"][index] = end

        self.memory_counter += 1

    def soft_update(self, TAU=0.01):
        # Store sample to replay buffer
        with torch.no_grad():
            for targetParam, evalParam in zip(self.critic_target.parameters(), self.critic.parameters()):
                targetParam.copy_((1 - self.tau) * targetParam.data + self.tau * evalParam.data)

    def learn(self):
        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        s_batch = [self.memory["s"][index] for index in sample_index]
        a_batch = [self.memory["a"][index] for index in sample_index]
        r_batch = [self.memory["r"][index] for index in sample_index]
        sn_batch = [self.memory["sn"][index] for index in sample_index]
        end_batch = [self.memory["end"][index] for index in sample_index]

        # TODO(Lab-04): Construct torch tensor
        ########################################################################################
        s_ts = torch.FloatTensor(np.array(s_batch)).to(device)
        a_ts = torch.FloatTensor(np.array(a_batch)).to(device)
        r_ts = torch.FloatTensor(np.array(r_batch)).to(device).view(self.batch_size, 1)
        sn_ts = torch.FloatTensor(np.array(sn_batch)).to(device)
        end_ts = torch.FloatTensor(np.array(end_batch)).to(device).view(self.batch_size, 1)
        ########################################################################################

        # TODO(Lab-05): Compute critic loss and optimize
        ########################################################################################
        # TD-target
        with torch.no_grad():
            a_next = self.actor(sn_ts)
            q_next_target = self.critic_target(sn_ts, a_next)
            q_target = r_ts + end_ts * self.gamma * q_next_target

        # Critic loss
        q_eval = self.critic(s_ts, a_ts)
        self.critic_loss = self.criterion(q_eval, q_target)

        self.critic_optim.zero_grad()
        self.critic_loss.backward()
        self.critic_optim.step()
        ########################################################################################

        # TODO(Lab-06): Compute actor loss and optimize
        ########################################################################################
        # Actor loss
        a_curr = self.actor(s_ts)
        q_current = self.critic(s_ts, a_curr)
        self.actor_loss = -q_current.mean()

        self.actor_optim.zero_grad()
        self.actor_loss.backward()
        self.actor_optim.step()
        ########################################################################################

        # TODO(Lab-07): Update target network and epsilon noise
        ########################################################################################
        self.soft_update()
        if self.epsilon > self.epsilon_params[1]:
            self.epsilon -= self.epsilon_params[2]
        else:
            self.epsilon = self.epsilon_params[1]
        ########################################################################################
        return float(self.actor_loss.detach().cpu().numpy()), float(self.critic_loss.detach().cpu().numpy())
