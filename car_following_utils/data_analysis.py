import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import matplotlib
matplotlib.use('agg')

def load_data(csv_dict):
    front_car_csv = pd.read_csv(csv_dict['frontCar'])
    following_car_csv = pd.read_csv(csv_dict['followingCar'])
    delta_s = np.array(following_car_csv['dhw'].tolist())

    front_car_v = np.array(front_car_csv['xVelocity'].to_list())
    following_car_v = np.array(following_car_csv['xVelocity'].to_list())

    following_car_a = np.array(following_car_csv['xAcceleration'].to_list())
    delta_v = front_car_v - following_car_v
    # 计算周围车辆与自车的关系
    following_car_x = np.array(following_car_csv['x'].tolist())

    surround_csv = pd.read_csv(csv_dict['surround'])
    left_preceding_x = np.array(surround_csv['leftPrecedingX'].tolist())
    left_preceding_dx = left_preceding_x - following_car_x
    left_preceding_v =np.array(surround_csv['leftPrecedingXVelocity'].tolist())
    left_preceding_dv = left_preceding_v - following_car_v

    left_alongside_x = np.array(surround_csv['leftAlongsideX'].tolist())
    left_alongside_dx = left_alongside_x - following_car_x
    left_alongside_v = np.array(surround_csv['leftAlongsideXVelocity'].tolist())
    left_alongside_dv = left_alongside_v - following_car_v

    right_preceding_x = np.array(surround_csv['rightPrecedingX'].tolist())
    right_preceding_dx = right_preceding_x - following_car_x
    right_preceding_v = np.array(surround_csv['rightPrecedingXVelocity'].tolist())
    right_preceding_dv = right_preceding_v - following_car_v

    rigth_alongside_x = np.array(surround_csv['rightAlongsideX'].tolist())
    right_alongside_dx = rigth_alongside_x - following_car_x
    right_alongside_v = np.array(surround_csv['rightAlongsideXVelocity'].tolist())
    right_alongside_dv = right_alongside_v - following_car_v
    return (delta_s, delta_v, following_car_v, following_car_a,
            [left_preceding_dx, left_preceding_dv, left_alongside_dx, left_alongside_dv, right_preceding_dx, right_preceding_dv, right_alongside_dx, right_alongside_dv])



def sub_csv(folder_path):
    sub_file = os.listdir(folder_path)
    sub_pkl = []
    for s in sub_file:
        file_class = s.split('.')[-1]
        if file_class == 'csv':
            sub_pkl.append(s)

    sub_csv_path = dict()

    for c in sub_pkl:
        name = c.split('.')[0]
        if name == 'front_car':
            sub_csv_path['frontCar'] = f'{folder_path}/{c}'
        elif name == 'following_car':
            sub_csv_path['followingCar'] = f'{folder_path}/{c}'
        elif name == 'aggregate':
            sub_csv_path['surround'] = f'{folder_path}/{c}'

    return sub_csv_path


def sub_folder(folder_path):
    sub_folders = os.listdir(folder_path)
    return [f'./{folder_path}/{s}' for s in sub_folders]


def main():
    data_info = {}
    time_len = 15
    datas_path = f'HighD_train_data/following_scene_withside_aggregate_{time_len}'
    sub_folders = sub_folder(datas_path)
    a = np.array([])
    r_s = np.array([])
    r_v = np.array([])
    f_v = np.array([])
    for s in tqdm(sub_folders):
        csv_dict = sub_csv(s)
        relative_s, relative_v, following_car_v, following_car_a, surround_datas = load_data(csv_dict)
        a = np.append(a, following_car_a)
        r_s = np.append(r_s, relative_s)
        r_v = np.append(r_v, relative_v)
        f_v = np.append(f_v, following_car_v)
    total_frame = len(a)
    total_sample = len(sub_folders)

    data_info['max_relative_s'] = np.max(r_s)
    data_info['min_relative_s'] = np.min(r_s)
    data_info['max_relative_v'] = np.max(r_v)
    data_info['min_relative_v'] = np.min(r_v)
    data_info['max_following_v'] = np.max(f_v)
    data_info['min_following_v'] = np.min(f_v)

    data_info['q1_relative_s'] = np.percentile(r_s, 25)
    data_info['q3_relative_s'] = np.percentile(r_s, 75)
    data_info['q1_relative_v'] = np.percentile(r_v, 25)
    data_info['q3_relative_v'] = np.percentile(r_v, 75)
    data_info['q1_following_v'] = np.percentile(f_v, 25)
    data_info['q3_following_v'] = np.percentile(f_v, 75)
    data_info['total_frame'] = total_frame
    data_info['total_sample'] = total_sample

    # 保存data_info
    with open(f'HighD_train_data/data_info_{time_len}.json', 'w') as f:
        json.dump(data_info, f, indent=4)
    data_info_path = f'./data_info/{time_len}'
    if not os.path.exists(data_info_path):
        os.makedirs(data_info_path)

    # 绘制f_v的概率密度图
    plt.figure()
    plt.hist(f_v, bins=150, color='#82B0D2',density=True)
    plt.ylabel('Probability Density')
    plt.xlabel('Following Car Velocity (m/s)')
    plt.grid(True)
    plt.savefig(f'{data_info_path}/following_v.png',dpi=500)
    plt.close()

    # 绘制r_s的概率密度图
    plt.figure()
    plt.hist(r_s, bins=150, color='#82B0D2',density=True)
    plt.ylabel('Probability Density')
    plt.xlabel('Relative Distance (m)')
    plt.grid(True)
    plt.savefig(f'{data_info_path}/relative_s.png',dpi=500)
    plt.close()

    # 绘制r_v的概率密度图
    plt.figure()
    plt.hist(r_v, bins=150, color='#82B0D2',density=True)
    plt.ylabel('Probability Density')
    plt.xlabel('Relative Velocity (m/s)')
    plt.grid(True)
    plt.savefig(f'{data_info_path}/relative_v.png',dpi=500)
    plt.close()

    # 绘制a的概率密度图
    plt.figure()

    plt.hist(a, bins=150, color='#82B0D2',density=True)
    plt.xlim(-3,3)
    plt.ylabel('Probability Density')
    plt.xlabel('Acceleration (m/s^2)')
    # 打开网格
    plt.grid(True)
    plt.savefig(f'{data_info_path}/acc.png',dpi=500)

    print(1)

if __name__ == "__main__":
    main()