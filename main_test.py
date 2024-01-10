import argparse
import numpy as np
import torch
from tqdm import tqdm
import csv
from HitLML_PPO_RLHF.agent import TestingAgent
from HitLML_PPO_RLHF.plotting_functions import plot_2d
from environment import Env
import os
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a PPO+RLHF agent for the CarRacing-v2')

    parser.add_argument('--render', default=True, action='store_true', help='render the environment')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--test_results_csv_folder', type=str, default='test_results/', help='Folder to save results')

    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)
    #--------------------Test--------------------
    torch.manual_seed(args.seed+1)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print('Testing Started....')
    parameter_dict_list = []
    for i,name in enumerate(os.listdir('./trained_parameters')):
        print(f'Testing with {name}')
        values = name.split('_')
        param_dict = {values[i]: values[i + 1] for i in range(0, len(values), 2)}
        parameter_dict_list.append(param_dict)
        img_stack = int(param_dict['rolloutsteps'])
        agent = TestingAgent(img_stack,args.device)
        folder = 'test_results/'
        name1='trained_parameters/'+name
        agent.load_param(name1)
        test_env = Env(img_stack, 1)

        training_records = []
        running_score = 0
        state = test_env.reset()
        filename = folder + name[:-4] + '.csv'
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

        #--------------------Plotting--------------------

        plot_2d(args.test_results_csv_folder, parameter_dict_list)