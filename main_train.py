import argparse
import numpy as np
import torch
from tqdm import tqdm
import csv
from datetime import datetime

from HitLML_PPO_RLHF.agent import TrainingAgent
from automated_labeler import HumanLabeler
from environment import Env

def rollout(env,agent,num_trajectories,init_state,rollout_steps = 4):
    """
    Rollout for one trajectory
    """
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

    return all_rollout_rewards
def rlhf_mode(env, agent, num_trajectories,init_state, device,rollout_steps,combo = 2):
    human_labeler = HumanLabeler(device)

    all_rollout_rewards = rollout(env, agent,num_trajectories,init_state,rollout_steps)
    all_combos = torch.combinations(all_rollout_rewards, combo).to(device)
    human_probs = human_labeler.label(all_combos).to(device)
    agent_probs = agent.compare_trajectories(all_combos).to(device)
        #best to worst
    loss = agent.net.rlhf_loss(human_probs,agent_probs)

    return loss, torch.max(all_rollout_rewards).item()
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a PPO+RLHF agent for the CarRacing-v0')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
    parser.add_argument('--action-repeat', type=int, default=8, metavar='N',
                        help='repeat action in N frames (default: 8)')
    parser.add_argument('--seed', type=int, default=0, metavar='N', help='random seed (default: 0)')
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
    parser.add_argument('--folder', type=str, default='training_results/', help='Folder to save results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--use_rlhf', type=bool, default=True, help='Use RLHF')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_trajectories', type=int, default=2, help='Number of Trajectories to compare between')
    parser.add_argument('--test_results_csv_folder', type=str, default='files_for_plotting/', help='Folder to save results')
    args = parser.parse_args()
    transition = np.dtype([('s', np.float64, (args.rollout_steps, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
                           ('r', np.float64), ('s_', np.float64, (args.rollout_steps, 96, 96))])
    agent = TrainingAgent(args.rollout_steps, args.gamma,args.device,transition,  args.rlhf_steps,args.num_rollouts,args.num_trajectories,args.use_rlhf,args.batch_size)
    env = Env(args.rollout_steps, args.action_repeat)

    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    filename_rlhf = args.folder+(f'summary_rlhf_img_stack_{args.rollout_steps}_num_epochs_{args.num_epochs}_rlhf_steps_{args.rlhf_steps}'
                                 f'num_rollouts_{args.num_rollouts}_rollout_steps_{args.rollout_steps}_rlhf_{args.use_rlhf}_{time_stamp}_traj_{args.num_trajectories}.csv')

    f_rlhf = open(filename_rlhf, "a")
    writer_rlhf = csv.DictWriter(f_rlhf, fieldnames=["epoch","Loss", 'Reward'])
    writer_rlhf.writeheader()
    csv_writer_rlhf = csv.writer(f_rlhf, delimiter=',')

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    training_records = []
    running_score = 0
    state = env.reset()
    print('Training Started....')
    name = (f'numepochs_{args.num_epochs}_rlhfsteps_{args.rlhf_steps}_numrollouts_'
                f'{args.num_rollouts}_rolloutsteps_{args.rollout_steps}_rlhf_{args.use_rlhf}_numtraj_{args.num_trajectories}')
    print(f'Parameters saved as {name}')
    for i_ep in tqdm(range(args.num_epochs)):

        score = 0
        state = env.reset()
        reward_for_epoch = []
        loss = 100
        for t in range(args.num_states):
            action, a_logp = agent.select_action(state)
            state_, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            reward_for_epoch.append(reward)

            if agent.store((state, action, a_logp, reward, state_)):
                print('updating')
                loss = agent.update()
            score += reward
            state = state_
            if done or die:
                break
        running_score = running_score * 0.99 + score * 0.01
        epoch_av_reward = np.mean(np.array(reward_for_epoch))
        csv_writer_rlhf.writerow([i_ep, loss, epoch_av_reward])

        if i_ep % args.log_interval == 0:
            print('Ep {}\tLast score: {:.2f}\tMoving average score: {:.2f}'.format(i_ep, score, running_score))

        if running_score > env.reward_threshold:
            print("Solved! Running reward is now {} and the last episode runs to {}!".format(running_score, score))
            break

    agent.save_param(name)

    f_rlhf.close()

