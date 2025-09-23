import os
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
import json


def sub_folder(folder_path, tra_len):
    sub_folders = os.listdir(folder_path)
    return [f'./HighD_train_data/following_scene_withside_aggregate_{tra_len}/{s}' for s in sub_folders]

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

def splitting(delta_T, status_len, relative_s, relative_v, following_car_v, following_car_a, sub_folder_path,sub_folder_name, surround_datas):
    left_preceding_dx, left_preceding_dv, left_alongside_dx, left_alongside_dv, right_preceding_dx, right_preceding_dv, right_alongside_dx, right_alongside_dv =\
        surround_datas[0], surround_datas[1], surround_datas[2], surround_datas[3], surround_datas[4], surround_datas[5], surround_datas[6], surround_datas[7]
    interval = round(delta_T / 0.04) # 每几个数据间隔提取新数据， 1代表不间隔， 2代表间隔1个
    indices = np.arange(0, len(relative_s), interval)
    output_len_T = 1
    input_len_T = status_len
    interval_relative_s = relative_s[indices]
    interval_relative_v = relative_v[indices]
    interval_following_v = following_car_v[indices]
    interval_following_a = following_car_a[indices]

    seq_relative_s = np.array(
        [interval_relative_s[i: i + input_len_T] for i in range(len(interval_relative_s) - input_len_T + 1)])
    seq_relative_v = np.array(
        [interval_relative_v[i: i + input_len_T] for i in range(len(interval_relative_v) - input_len_T + 1)])
    seq_following_v = np.array(
        [interval_following_v[i: i + input_len_T] for i in range(len(interval_following_v) - input_len_T + 1)])

    target_following_a = np.array([interval_following_a[i: i + output_len_T] for i in
                                   range(input_len_T - 1, len(interval_following_a) - output_len_T + 1)])

    data_json={}
    for i in range(len(target_following_a)):
        sample_one = {
            'seq_relative_s': seq_relative_s[i],
            'seq_relative_v': seq_relative_v[i],
            'seq_following_v': seq_following_v[i],
            'target_following_a': target_following_a[i],
        }
        with open(f"{sub_folder_path}/{i}.pkl", 'wb') as f:
            pickle.dump(sample_one, f)
        data_json[f"{sub_folder_name}_{i}"] = f'{sub_folder_path}/{i}.pkl'
    return data_json


def individualize(path_list, set_class, up_folder_path, delta_T, status_len):
    data_save = {}
    for s in tqdm(path_list):
        sub_folder_name = s.split('/')[-1]
        sub_folder_path = f'{up_folder_path}/{set_class}/{sub_folder_name}'
        if not os.path.exists(sub_folder_path):
            os.makedirs(sub_folder_path)  # 递归创建多级目录

        csv_dict = sub_csv(s)
        relative_s, relative_v, following_car_v, following_car_a, surround_datas = load_data(csv_dict)

        sub_path = splitting(delta_T, status_len, relative_s, relative_v, following_car_v, following_car_a, sub_folder_path,sub_folder_name, surround_datas)
        data_save.update(sub_path)
    with open(f"./{up_folder_path}/{set_class}/{delta_T}_{status_len}_seq2one.json", "w", encoding="utf-8") as f:
        json.dump(data_save, f, ensure_ascii=False, indent=4)


def split_train_val(path_list):
    with open(f"./HighD_train_data/test_list.json", "r", encoding="utf-8") as f:
        test_json = json.load(f)
    test_list = list(test_json.keys())#[:486]
    train_val_path = []
    test_path = []
    for p in path_list:
        file_name = p.split('/')[-1]
        temp = file_name.split('_')[:3]
        temp = '_'.join(temp)
        if temp not in test_list:
            train_val_path.append(p)
        else:
            test_path.append(p)
    random.shuffle(train_val_path)
    val_list = train_val_path[:1110]
    # todo: 将train_val_path按比例或数量进行分开
    train_list = train_val_path[1110:]
    # test_list = test_path

    return train_list, val_list

def main():
    #-------------------超参数-------------------
    tra_len = 15  # 一般不用改
    random.seed(42)
    delta_T = 0.08 # 每个数据间隔
    #-------------------------------------------
    status_len = 8
    status_len = round(status_len / delta_T) # 状态长度

    up_folder_path = f'./Splitting_data/{delta_T}_{status_len}_seq2one'
    if not os.path.exists(up_folder_path):
        os.makedirs(up_folder_path)  # 递归创建多级目录
    data_save_train = {}
    data_save_test = {}

    datas_path = f'./HighD_train_data/following_scene_withside_aggregate_{tra_len}'
    sub_folders = sub_folder(datas_path, tra_len)

    train_list, val_list = split_train_val(sub_folders)


    # random.shuffle(sub_folders)
    # # 将sub_folders分为70%,15%,15%
    # split_index_70 = int(len(sub_folders) * 0.7)
    # split_index_85 = int(len(sub_folders) * 0.85)
    # list_70 = sub_folders[:split_index_70]
    # list_85 = sub_folders[split_index_70:split_index_85]
    # list_15 = sub_folders[split_index_85:]

    sample_dict = {
        'train_list' : [i.split('/')[-1] for i in train_list],
        'val_list' : [i.split('/')[-1] for i in val_list],
    }

    individualize(train_list, 'train',up_folder_path, delta_T, status_len)
    individualize(val_list, 'val',up_folder_path, delta_T, status_len)
    # individualize(list_15, 'test',up_folder_path, delta_T, status_len)

    with open(f"./{up_folder_path}/set_group.json", "w", encoding="utf-8") as f:
        json.dump(sample_dict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()