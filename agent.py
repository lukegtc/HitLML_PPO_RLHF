import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tqdm import tqdm
from automated_labeler import HumanLabeler
from conv_network import Net

class TrainingAgent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss

    def __init__(self,img_stack,gamma,device,transition, ppo_epoch,buffer_capacity,num_trajectories,rlhf,batch_size=128):
        self.training_step = 0
        self.device = device
        self.net = Net(img_stack).double().to(self.device)
        self.transition = transition
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.gamma = gamma
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)
        self.rlhf_optimizer = optim.Adam(self.net.parameters(), lr=7e-3, weight_decay=1e-5)
        self.rlhf_loss = nn.CrossEntropyLoss()
        self.num_trajectories = num_trajectories
        self.rlhf = rlhf


    def select_action(self, state):
        if state.dtype != torch.double:
            state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)

        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp
    def select_action_with_update(self, state):
        if state.dtype != torch.double:
            state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        alpha, beta = self.net(state)[0]
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def save_param(self,extension):
        torch.save(self.net.state_dict(), f'trained_parameters/{extension}.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1


        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        human = HumanLabeler(self.device)

        self.training_step += 1

        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)

        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r.view(-1, 1) + self.gamma * self.net(s_)[1]
            adv = target_v - self.net(s)[1]
        losses_per_epoch = []
        for _ in tqdm(range(self.ppo_epoch)):
            all_losses = []
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):

                alpha, beta = self.net(s[index])[0]
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])

                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.net(s[index])[1], target_v[index])
                if self.rlhf:
                    combinations = torch.combinations(r[index], self.num_trajectories)
                    human_probs = human.label(combinations).to(self.device)
                    agent_probs = self.compare_trajectories(combinations).to(self.device)
                    rlhf_loss = self.net.rlhf_loss(human_probs, agent_probs)
                    # rlhf_loss, best_traj_reward = rlhf_mode(env, self, num_trajectories, s[index][-1],device=self.device,rollout_steps=4,combo = 2)
                    loss = action_loss + 2. * value_loss + 10*rlhf_loss
                else:
                    loss = action_loss + 2. * value_loss
                all_losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            vals = torch.stack(all_losses)
            vals = torch.mean(vals)
            losses_per_epoch.append(vals)

        return torch.mean(torch.stack(losses_per_epoch))

    def compare_trajectories(self, rewards):

        prob_sigma = torch.nn.functional.softmax(rewards, dim=1)

        return prob_sigma

class TestingAgent():
    """
    Agent for testing
    """
    def __init__(self, img_stack,device):
        self.device = device
        self.net = Net(img_stack).float().to(self.device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self,name):
        self.net.load_state_dict(torch.load(name))
