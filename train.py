import argparse

import numpy as np
import math
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import cv2
from tqdm import tqdm
import random
import itertools
import csv
from datetime import datetime



class Env():
    """
    Environment wrapper for CarRacing 
    """

    def __init__(self,img_stack, action_repeat):
        self.env = gym.make('CarRacing-v2', domain_randomize=True)
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat


    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_gray = self.rgb2gray(img_rgb[0])
        self.stack = [img_gray] * self.img_stack # four frames for decision
        return np.array(self.stack)

    def step(self, action):
        total_reward = 0
        for i in range(self.action_repeat):
            img_rgb, reward, die, _,_ = self.env.step(action)
            # don't penalize "die state"
            if die:
                reward += 100
            # green penalty
            if np.mean(img_rgb[:, :, 1]) > 185.0:
                reward -= 0.05
            total_reward += reward
            # if no reward recently, end the episode
            done = True if self.av_r(reward) <= -0.1 else False
            if done or die:
                break
        img_gray = self.rgb2gray(img_rgb)
        self.stack.pop(0)
        self.stack.append(img_gray)
        assert len(self.stack) == self.img_stack
        return np.array(self.stack), total_reward, done, die

    def render(self, *arg):
        self.env.render(*arg)



    @staticmethod
    def rgb2gray(state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        state /= 255.0
        return state
    @staticmethod
    def reward_memory():
        # record reward for last 100 steps
        count = 0
        length = 100
        history = np.zeros(length)

        def memory(reward):
            nonlocal count
            history[count] = reward
            count = (count + 1) % length
            return np.mean(history)

        return memory


class Net(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self,img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.LeakyReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.LeakyReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.LeakyReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.LeakyReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.LeakyReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.LeakyReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)
        self.rlhf_loss = nn.CrossEntropyLoss()

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1

        return (alpha, beta), v
class HumanLabeler():

    def __init__(self,device):
        self.device = device
        return

    def label(self, rewards):


        noisy_rewards = rewards + torch.randn(rewards.shape).to(self.device)*rewards*0.1


        human_prob = torch.nn.functional.softmax(noisy_rewards, dim=1)
        # human_prob = 1.
        if random.random() <= 0.1:
            human_prob = torch.nn.functional.normalize(torch.rand(rewards.shape))
            human_prob = torch.softmax(human_prob, dim=1)
        return human_prob

def rollout(env,agent,num_trajectories,init_state,rollout_steps = 4):
    """
    Rollout for one trajectory
    """
    # trajectory = {"states": [0]*rollout_steps, "actions": [0]*rollout_steps, "rewards": [0]*rollout_steps, "a_logps": [0]*rollout_steps}
    # state = init_state

    all_rollout_rewards = torch.zeros(num_trajectories)
    for trajectory in range(num_trajectories):
        reward_total_per_rollout = 0
        state = init_state
        for i in range(rollout_steps):
            action, a_logp = agent.select_action_with_update(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            reward_total_per_rollout += reward
            state = state_

        all_rollout_rewards[trajectory] = reward_total_per_rollout


    # return torch.sum(torch.tensor(rewards, requires_grad=True).to(device))
    return all_rollout_rewards
def rlhf_mode(env, agent, num_trajectories,init_state, device,rollout_steps,combo = 2):
    human_labeler = HumanLabeler(device)

    all_rollout_rewards = rollout(env, agent,num_trajectories,init_state,rollout_steps)
    all_combos = torch.combinations(all_rollout_rewards, combo)
    human_probs = human_labeler.label(all_combos)
    agent_probs = agent.compare_trajectories(all_combos)
        #best to worst
    loss = agent.net.rlhf_loss(human_probs,agent_probs)

    return loss, torch.max(all_rollout_rewards).item()

class Agent():
    """
    Agent for training
    """
    max_grad_norm = 0.5
    clip_param = 0.1  # epsilon in clipped loss

    def __init__(self,img_stack,gamma,device,transition, ppo_epoch,buffer_capacity,batch_size=128):
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


    # def rlhf_loss(self, probs, human_probs):
    #
    #     positives = human_probs[0]*torch.log(probs[0]+1e-6)
    #     negatives = human_probs[1]*torch.log(probs[1]+1e-6)
    #
    #     loss = -torch.sum(positives + negatives)
    #     return loss

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
        torch.save(self.net.state_dict(), f'param_final/ppo_net_params_{extension}_final.pkl')

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1


        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self,num_trajectories,rlhf = False):
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
            # adv = (adv - adv.mean()) / (adv.std() + 1e-8)
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
                if rlhf:
                    combinations = torch.combinations(r[index], num_trajectories)
                    human_probs = human.label(combinations).to(self.device)
                    agent_probs = self.compare_trajectories(combinations).to(self.device)
                    rlhf_loss = agent.net.rlhf_loss(human_probs, agent_probs)
                    # rlhf_loss, best_traj_reward = rlhf_mode(env, self, num_trajectories, s[index][-1],device=self.device,rollout_steps=rollout_steps,combo = 2)
                    loss = action_loss + 2. * value_loss + 10*rlhf_loss
                else:
                    loss = action_loss + 2. * value_loss
                all_losses.append(loss)
                self.optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
            vals = torch.stack(all_losses)
            vals = torch.mean(vals)
            losses_per_epoch.append(vals)

        return torch.mean(torch.stack(losses_per_epoch))

    def compare_trajectories(self, rewards):

        prob_sigma = torch.nn.functional.softmax(rewards, dim=1)

        return prob_sigma


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description='Train a PPO agent for the CarRacing-v0')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--action-repeat', type=int, default=8, metavar='N',
                        help='repeat action in N frames (default: 8)')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument(
        '--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1)')
    parser.add_argument('--num-states', type=int, default=200,
                        help='number of states to train (default: 1)')
    parser.add_argument('--rlhf-steps', type=int, default=200,
                        help='number of rlhf steps to train (default: 1)')
    parser.add_argument('--num-rollouts', type=int, default=100, help='Number of rollouts')
    parser.add_argument('--rollout-steps', type=int, default=4, help='Number of rollout steps')
    parser.add_argument('--folder', type=str, default='final_results/', help='Folder to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--use_rlhf', type=bool, default=False, help='Use RLHF')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_trajectories', type=int, default=2, help='Number of Trajectories to compare between')
    args = parser.parse_args()
    transition = np.dtype([('s', np.float64, (args.rollout_steps, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                           ('r', np.float64), ('s_', np.float64, (args.rollout_steps, 96, 96))])
    agent = Agent(args.rollout_steps, args.gamma,args.device,transition,  args.rlhf_steps,args.num_rollouts,args.batch_size)
    env = Env(args.rollout_steps, args.action_repeat)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if not args.use_rlhf:
        filename_ppo = args.folder+f'summary_ppo_img_stack_{args.rollout_steps}_num_epochs_{args.num_epochs}__rlhf_{args.use_rlhf}_{time_stamp}.csv'
    else:
        filename_ppo = args.folder + (
            f'summary_rlhf_ppo_img_stack_{args.rollout_steps}_num_epochs_{args.num_epochs}_rlhf_steps_'
            f'num_rollouts_{args.num_rollouts}_rollout_steps_{args.rollout_steps}_rlhf_{args.use_rlhf}_traj_{args.num_trajectories}.csv')
    filename_rlhf = args.folder+(f'summary_rlhf_rewards_only_img_stack_{args.rollout_steps}_num_epochs_{args.num_epochs}_rlhf_steps_{args.rlhf_steps}'
                                 f'num_rollouts_{args.num_rollouts}_rollout_steps_{args.rollout_steps}_rlhf_{args.use_rlhf}_new_human_func_{time_stamp}_traj_{args.num_trajectories}.csv')
    f_ppo = open(filename_ppo, "a")
    f_rlhf = open(filename_rlhf, "a")
    writer_ppo = csv.DictWriter(f_ppo, fieldnames=["epoch","Loss", 'Reward Average'])
    writer_ppo.writeheader()
    writer_rlhf = csv.DictWriter(f_rlhf, fieldnames=["Loss", 'Reward'])
    writer_rlhf.writeheader()
    print(f'Created {filename_ppo}...')
    print(f'Created {filename_rlhf}...')

    csv_writer_ppo = csv.writer(f_ppo, delimiter=',')
    csv_writer_rlhf = csv.writer(f_rlhf, delimiter=',')

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    training_records = []
    running_score = 0
    state = env.reset()
    for i_ep in tqdm(range(args.num_epochs)):


        score = 0
        state = env.reset()
        reward_for_epoch = []
        loss = 100
        for t in tqdm(range(args.num_states)):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            reward_for_epoch.append(reward)
            if args.render:
                env.render(render_mode='human')
            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                loss = agent.update(num_trajectories = args.num_trajectories, rlhf = args.use_rlhf)
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        epoch_av_reward = np.mean(np.array(reward_for_epoch))
        csv_writer_ppo.writerow([i_ep, loss, epoch_av_reward])
        csv_writer_rlhf.writerow([i_ep, loss, epoch_av_reward])

        if i_ep % args.log_interval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))

        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break
    name = (f'ppo_img_stack_{args.rollout_steps}_num_epochs_{args.num_epochs}_rlhf_steps_{args.rlhf_steps}_num_rollouts_'
                f'{args.num_rollouts}_rollout_steps_{args.rollout_steps}_rlhf_{args.use_rlhf}_num_traj_{args.num_trajectories}')
    agent.save_param(name)

    # if args.use_rlhf:
    #     print('starting RLHF')
    #     agent.net.train()
    #     for j in tqdm(range(args.rlhf_steps)):
    #         rlhf_loss, rlhf_rollout_reward_avg = rlhf(env, agent, args.num_rollouts, state, args.rollout_steps, combo = 2)
    #         test = list(agent.net.parameters())[0]
    #         if rlhf_rollout_reward_avg > previous_rlhf_reward:
    #             agent.save_param(f'rlhf_{name}_with_rlhf_new_human_func_reward_{rlhf_rollout_reward_avg}')
    #             previous_rlhf_reward = rlhf_rollout_reward_avg
    #         print('rlhf_loss: ', rlhf_loss.item())
    #         print('rlhf_rollout_reward: ', rlhf_rollout_reward_avg)
    #         csv_writer_rlhf.writerow([rlhf_loss, rlhf_rollout_reward_avg])
    # else:
    #     rlhf_loss = 0
    #     rlhf_rollout_reward = 0

    f_ppo.close()
    f_rlhf.close()