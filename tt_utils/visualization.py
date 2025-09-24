import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

def plot_trajectory(ego_feature, agent_feature):
    T = ego_feature['position'].shape[0]
    save_path = './visualization'
    os.makedirs(save_path, exist_ok=True)

    for t in range(T):
        fig, ax = plt.subplots(figsize=(50, 50))
        now_position = ego_feature['position'][t]
        now_shape = ego_feature['shape'][t]
        # 以now_position为矩形中点，now_shape为矩形的长和宽，画图
        rectangle = plt.Rectangle((now_position[0] - now_shape[0] / 2, now_position[1] - now_shape[1] / 2),
                                  now_shape[0], now_shape[1], fc='blue', ec='black', alpha=0.5)
        ax.add_patch(rectangle)
        N = agent_feature['position'].shape[0]
        for n in range(N):
            if agent_feature['valid_mask'][n, t] == 0:
                continue
            agent_position = agent_feature['position'][n, t]
            agent_shape = agent_feature['shape'][n, t]
            rectangle = plt.Rectangle((agent_position[0] - agent_shape[0] / 2, agent_position[1] - agent_shape[1] / 2),
                                      agent_shape[0], agent_shape[1], fc='red', ec='black', alpha=0.5)
            ax.add_patch(rectangle)
        ax.set_xlim(now_position[0] - 100, now_position[0] + 100)
        ax.set_ylim(now_position[1] - 100, now_position[1] + 100)
        plt.savefig(f'{save_path}/{t:03d}.png')
        plt.close()

        print(1)

def main():
    data_root = "../processed_data"
    # 查看data_root文件夹下的所有文件
    files = os.listdir(data_root)

    files_path = data_root + "/scenario_01/car_12/51.pkl"
    with open(files_path, 'rb') as f:
        data = pickle.load(f)
    ego_feature = data["ego_feature"]
    agent_feature = data["agent_feature"]
    current_state = data["current_state"]
    plot_trajectory(ego_feature, agent_feature)
    print(1)


if __name__ == "__main__":
    main()