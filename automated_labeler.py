import torch
import random
import numpy as np
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