import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re


def plot_2d(csv_folder):
    dict_of_data = {}
    for file in os.listdir(csv_folder):
        f = open(csv_folder+'/'+file, 'r')
        table = np.array(list(csv.reader(f, delimiter=',')))

        # name = re.findall(r"loss.{1,}", file)
        # name = name[0].replace('.csv', '')
        name=file
        table_dict = {}
        for i in range(len(table[0])):
            table_dict[table[0][i]] = np.array(table[1:, i],dtype=np.float32)
        dict_of_data[name] = table_dict

        f.close()
    diff = []
    statistics_of_run={'mean':[],'std':[]}
    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()

    idx = np.arange(len(dict_of_data.keys()))
    i=0
    for key in dict_of_data.keys():
        print(key)
        reward = dict_of_data[key]['Reward']
        print(np.mean(reward))
        print(np.std(reward))


        statistics_of_run['mean'].append({key:np.mean(reward)})
        statistics_of_run['std'].append({key:np.std(reward)})
        name = key.split('_')[10:]
        # num_epochs = int(name[2])
        # if name[5] is not '20' or name[5] is not '10':
        #     num_rlhf_steps = 10
        # else:
        # num_rlhf_steps = int(name[8])
        # rollout_steps = int(name[14])
        #
        is_rlhf = name[16]
        num_traj_steps = int(name[19])
        is_rlhf = name[16]
        rollout_steps = int(name[14])
        num_rlhf_steps = int(name[8])
        #
        if is_rlhf == 'True':
            label = f'rollout_steps: {rollout_steps}, is_rlhf: {is_rlhf}, trajectories: {num_traj_steps}'
        else:
          label = f'rollout_steps: {rollout_steps}, trajectories: {num_traj_steps}'
        ax.errorbar(x=idx[i], y=np.mean(reward), yerr=np.std(reward), marker='s',label=label)
        # print(key)
        # print(np.mean(reward))
        #
        #
        #
        # diff.append(reward)
        ax1.scatter(np.arange(len(dict_of_data[key]['Reward'])),dict_of_data[key]['Reward'], label=label)
        i+=1
    # print(sum(diff[1]-diff[0])/len(diff[0]))
    # print(statistics_of_run)
    ax1.set_xlabel('Runs')
    ax1.set_ylabel('Reward')
    ax1.grid()
    ax1.legend(loc='upper right')
    ax1.set_title('Number of Trajectories')
    fig1.savefig('num_traj.png', bbox_inches='tight')
    fig1.show()
    ax.set_xlabel('Runs')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper right')
    ax.grid()
    ax.set_title('Number of Trajectories')
    fig.savefig('mean_std_num_traj.png', bbox_inches='tight')


    fig.show()

if __name__ == '__main__':
    plot_2d('/home/lukegtc/PycharmProjects/HLML/pytorch_car_caring-master/500_epochs/')
