import os
import pandas as pd
from tqdm import tqdm


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
        if name == 'frontCar':
            sub_csv_path['frontCar'] = f'{folder_path}/{c}'
        elif name == 'followingCar':
            sub_csv_path['followingCar'] = f'{folder_path}/{c}'
        elif name == 'left_alongside':
            sub_csv_path['left_alongside'] = f'{folder_path}/{c}'
        elif name == 'left_preceding':
            sub_csv_path['left_preceding'] = f'{folder_path}/{c}'
        elif name == 'right_alongside':
            sub_csv_path['right_alongside'] = f'{folder_path}/{c}'
        elif name == 'right_preceding':
            sub_csv_path['right_preceding'] = f'{folder_path}/{c}'

    return sub_csv_path


def sub_folder(folder_path, following_time):
    sub_folders = os.listdir(folder_path)
    return [f'./HighD_train_data/following_scene_withside_changed_direction_{following_time}/{s}' for s in sub_folders]


def aggregate_side(csv_dict):
    front_df = pd.read_csv(csv_dict['frontCar'])
    following_df = pd.read_csv(csv_dict['followingCar'])
    left_alongside_csv = pd.DataFrame(columns=following_df.columns)
    left_preceding_csv = pd.DataFrame(columns=following_df.columns)
    right_alongside_csv = pd.DataFrame(columns=following_df.columns)
    right_preceding_csv = pd.DataFrame(columns=following_df.columns)
    for file_name, file_path in csv_dict.items():
        if file_name == 'left_alongside':
            left_alongside_csv = pd.read_csv(file_path)
        elif file_name == 'left_preceding':
            left_preceding_csv = pd.read_csv(file_path)
        elif file_name == 'right_alongside':
            right_alongside_csv = pd.read_csv(file_path)
        elif file_name == 'right_preceding':
            right_preceding_csv = pd.read_csv(file_path)
    aggregate_df = pd.DataFrame(columns=["frame", "leftPrecedingId", "leftPrecedingX", "leftPrecedingXVelocity",
                                         "leftAlongsideId", "leftAlongsideX", "leftAlongsideXVelocity",
                                         "rightPrecedingId", "rightPrecedingX", "rightPrecedingXVelocity",
                                         "rightAlongsideId", "rightAlongsideX", "rightAlongsideXVelocity"])

    # 依次获取copy_following_df的每一行数据
    for i in range(following_df.shape[0]):
        new_line_dict = {}
        i_data = following_df.iloc[i]
        frame = i_data["frame"]
        new_line_dict["frame"] = frame
        # 处理left_preceding_df 👇

        if frame in left_preceding_csv["frame"].tolist():
            left_preceding_data = left_preceding_csv[left_preceding_csv["frame"] == frame]
            leftPrecedingId = left_preceding_data["id"].values[0]
            leftPrecedingX = left_preceding_data["x"].values[0]
            leftPrecedingXVelocity = left_preceding_data["xVelocity"].values[0]
            new_line_dict["leftPrecedingId"] = leftPrecedingId
            new_line_dict["leftPrecedingX"] = leftPrecedingX
            new_line_dict["leftPrecedingXVelocity"] = leftPrecedingXVelocity

        # 处理left_alongside_df 👇
        if frame in left_alongside_csv["frame"].tolist():
            left_alongside_data = left_alongside_csv[left_alongside_csv["frame"] == frame]
            leftAlongsideId = left_alongside_data["id"].values[0]
            leftAlongsideX = left_alongside_data["x"].values[0]
            leftAlongsideXVelocity = left_alongside_data["xVelocity"].values[0]
            new_line_dict["leftAlongsideId"] = leftAlongsideId
            new_line_dict["leftAlongsideX"] = leftAlongsideX
            new_line_dict["leftAlongsideXVelocity"] = leftAlongsideXVelocity

        # 处理right_preceding_df 👇
        if frame in right_preceding_csv["frame"].tolist():
            right_preceding_data = right_preceding_csv[right_preceding_csv["frame"] == frame]
            rightPrecedingId = right_preceding_data["id"].values[0]
            rightPrecedingX = right_preceding_data["x"].values[0]
            rightPrecedingXVelocity = right_preceding_data["xVelocity"].values[0]
            new_line_dict["rightPrecedingId"] = rightPrecedingId
            new_line_dict["rightPrecedingX"] = rightPrecedingX
            new_line_dict["rightPrecedingXVelocity"] = rightPrecedingXVelocity

        # 处理right_alongside_df 👇
        if frame in right_alongside_csv["frame"].tolist():
            right_alongside_data = right_alongside_csv[right_alongside_csv["frame"] == frame]
            rightAlongsideId = right_alongside_data["id"].values[0]
            rightAlongsideX = right_alongside_data["x"].values[0]
            rightAlongsideXVelocity = right_alongside_data["xVelocity"].values[0]
            new_line_dict["rightAlongsideId"] = rightAlongsideId
            new_line_dict["rightAlongsideX"] = rightAlongsideX
            new_line_dict["rightAlongsideXVelocity"] = rightAlongsideXVelocity


        new_line = pd.DataFrame(new_line_dict, index=[0])
        aggregate_df = pd.concat([aggregate_df, new_line], ignore_index=True)

    return aggregate_df, front_df, following_df



def surround_aggregate(following_time):
    data_root = f'./HighD_train_data/following_scene_withside_changed_direction_{following_time}/'
    sub_folders = sub_folder(data_root, following_time)
    for s in tqdm(sub_folders):
        folder_name = s.split('/')[-1]
        new_folder = f'./HighD_train_data/following_scene_withside_aggregate_{following_time}/{folder_name}'
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        csv_dict = sub_csv(s)
        aggregate_df, front_car_csv, following_car_csv = aggregate_side(csv_dict)
        # 分别保存front_car_csv, following_car_csv, aggregate_df
        front_car_csv.to_csv(f'{new_folder}/front_car.csv', index=False)
        following_car_csv.to_csv(f'{new_folder}/following_car.csv', index=False)
        aggregate_df.to_csv(f'{new_folder}/aggregate.csv', index=False)


