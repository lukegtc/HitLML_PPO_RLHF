import argparse

import numpy as np

import gymnasium as gym
import torch
import torch.nn as nn
import cv2
from tqdm import tqdm
import csv
import os
#https://github.com/xtma/pytorch_car_caring/blob/master/train.py#L23




class Env():
    """
    Test environment wrapper for CarRacing 
    """

    def __init__(self, img_stack, action_repeat):
        self.env = gym.make('CarRacing-v2',render_mode='human')
        self.reward_threshold = self.env.spec.reward_threshold
        self.img_stack = img_stack
        self.action_repeat = action_repeat

    def reset(self):
        self.counter = 0
        self.av_r = self.reward_memory()

        self.die = False
        img_rgb = self.env.reset()
        img_rgb = img_rgb[0]

        img_gray = self.rgb2gray(img_rgb)
        self.stack = [img_gray] * self.img_stack
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

    def render_me(self, render=True):
        self.env.render()

    @staticmethod
    def rgb2gray(state):
        state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        state = state.astype(float)
        state /= 255.0
        return state

    @staticmethod
    def reward_memory():
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
    def __init__(self, img_stack):
        super(Net, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(img_stack, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, 100), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, 100), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(100, 3), nn.Softplus())
        self.apply(self._weights_init)

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


class Agent():
    """
    Agent for testing
    """
    def __init__(self, img_stack):
        self.net = Net(img_stack).float().to(device)

    def select_action(self, state):
        state = torch.from_numpy(state).float().to(device).unsqueeze(0)
        with torch.no_grad():
            alpha, beta = self.net(state)[0]
        action = alpha / (alpha + beta)

        action = action.squeeze().cpu().numpy()
        return action

    def load_param(self,name):
        self.net.load_state_dict(torch.load(name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test the PPO agent for the CarRacing-v0')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--render',default=True, action='store_true', help='render the environment')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    for i,name in enumerate(os.listdir('./trained_parameters')):
        values = name.split('_')
        param_dict = {values[i]: values[i + 1] for i in range(0, len(values), 2)}
        img_stack = param_dict['imgstack']
        agent = Agent(img_stack)
        folder = 'test_results/'
        name1='trained_parameters/'+name
        agent.load_param(name1)
        test_env = Env(img_stack, 1)

        training_records = []
        running_score = 0
        state = test_env.reset()
        filename = folder + 'test_results' + name[:-4] + '.csv'
        f = open(filename, "a")
        writer_ppo = csv.DictWriter(f, fieldnames=["step", 'Reward'])
        writer_ppo.writeheader()
        print(f'Created {filename}...')
        csv_writer = csv.writer(f, delimiter=',')
        for i_ep in range(100):
            score = 0
            state = test_env.reset()

            for t in tqdm(range(500)):
                action = agent.select_action(state)
                state_, reward, done, die = test_env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
                if args.render:
                    test_env.render_me()
                score += reward
                state = state_
                if done or die:
                    break
            csv_writer.writerow([i_ep, score])

            print('Ep {}\tScore: {:.2f}\t'.format(i_ep, score))
        f.close()