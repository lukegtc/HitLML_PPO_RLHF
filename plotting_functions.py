import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re


def plot_2d(csv_folder, param_dict_list):
    dict_of_data = {}
    for i, file in enumerate(os.listdir(csv_folder)):
        f = open(csv_folder+'/'+file, 'r')
        param_dict = param_dict_list[i]
        table = np.array(list(csv.reader(f, delimiter=',')))

        name=file
        table_dict = {}
        for i in range(len(table[0])):
            table_dict[table[0][i]] = np.array(table[1:, i],dtype=np.float32)
        dict_of_data[name] = table_dict

        f.close()
    statistics_of_run={'mean':[],'std':[]}
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    idx = np.arange(len(dict_of_data.keys()))
    i=0
    for key in dict_of_data.keys():
        print(key)
        reward = dict_of_data[key]['Reward']
        print(f'Mean of all runs in this folder: {np.mean(reward)}')
        print(f'Standard Deviation of all runs in this folder: {np.std(reward)}')


        statistics_of_run['mean'].append({key:np.mean(reward)})
        statistics_of_run['std'].append({key:np.std(reward)})

        num_traj_steps = param_dict['rlhfsteps']
        is_rlhf = param_dict['rlhf']
        rollout_steps = param_dict['rolloutsteps']
        num_trajectories = param_dict['numtraj'][:-4]
        num_epochs = param_dict['numepochs']

        #
        if is_rlhf == 'True':
            label = f'rollout_steps: {rollout_steps}, is_rlhf: {is_rlhf}, trajectories: {num_traj_steps}, num_trajectories: {num_trajectories}'
        else:
          label = f'rollout_steps: {rollout_steps}, trajectories: {num_traj_steps}'
        ax.errorbar(x=idx[i], y=np.mean(reward), yerr=np.std(reward), marker='s',label=label)

        ax1.scatter(np.arange(len(dict_of_data[key]['Reward'])),dict_of_data[key]['Reward'], label=label)
        i+=1

    ax1.set_xlabel('Runs')
    ax1.set_ylabel('Reward')
    ax1.grid()
    ax1.legend(loc='upper right')
    ax1.set_title('Scores')
    fig1.savefig('./plotting/traj_scattered.png', bbox_inches='tight')
    fig1.show()
    ax.set_xlabel('Runs')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title('Mean and Deivation of Scores')
    fig.savefig(f'./plotting/mean_std_num_traj.png', bbox_inches='tight')


    fig.show()

# if __name__ == '__main__':
#     plot_2d('/home/lukegtc/PycharmProjects/HLML/pytorch_car_caring-master/files_for_plotting/')
