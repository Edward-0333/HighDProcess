import os
import pandas as pd
from tqdm import tqdm


def change_direction(file_path):
    following_df = pd.read_csv(file_path)
    copy_following_df = following_df.copy()
    # 获取following_df的drivingDirection
    following_drivingDirection = following_df["drivingDirection"][0]
    # 判断following_df的drivingDirection是否为1
    if following_drivingDirection == 1:
        following_df_average = (copy_following_df["frontSightDistance"] +
                                copy_following_df["backSightDistance"]).tolist()
        following_df_average = sum(following_df_average) / len(following_df_average)
        # 将following_df中的x坐标按following_df_average进行取反
        copy_following_df["x"] = following_df_average - copy_following_df["x"]
        # xVelocity取反
        copy_following_df["xVelocity"] = -copy_following_df["xVelocity"]
        # xAcceleration取反
        copy_following_df["xAcceleration"] = -copy_following_df["xAcceleration"]
        copy_following_df["precedingXVelocity"] = -copy_following_df["precedingXVelocity"]
    # 判断copy_following_df的xVelocity是否为正数, 若存在负数则把为负的数据设为0.01
    if not all(copy_following_df["xVelocity"] > 0):
        copy_following_df.loc[copy_following_df["xVelocity"] < 0, "xVelocity"] = 0.01
    # 判断copy_following_df的precedingXVelocity是否为正数, 若存在负数则把为负的数据设为0.01

    return copy_following_df


def process_proceeding(tracks_df, following_df, p, car_item, car, following_drivingDirection):
    # 获取tracks_df中id为p的车辆
    preceding_df = tracks_df[tracks_df["id"] == p]
    # 筛选出preceding_df中与car_df相同frame的数据
    preceding_df = preceding_df[preceding_df["frame"].isin(following_df["frame"].values)]
    # 判断preceding_df的行数是否与car_df的行数相等
    assert preceding_df.shape[0] == following_df.shape[0], "行数不相等"
    # 判断car_item中id为p的drivingDirection是否与following_drivingDirection相等
    assert (car_item[car_item["id"] == p]["drivingDirection"].values[0] == following_drivingDirection), "drivingDirection不相等"
    # 根据following_drivingDirection修改preceding_df中的数据
    copy_preceding_df = preceding_df.copy()
    if following_drivingDirection == 1:
        preceding_df_average = (copy_preceding_df["frontSightDistance"] +
                                copy_preceding_df["backSightDistance"]).tolist()
        preceding_df_average = sum(preceding_df_average) / len(preceding_df_average)
        # 将preceding_df中的x坐标按preceding_df_average进行取反
        copy_preceding_df["x"] = preceding_df_average - copy_preceding_df["x"]
        # xVelocity取反
        copy_preceding_df["xVelocity"] = -copy_preceding_df["xVelocity"]
        # xAcceleration取反
        copy_preceding_df["xAcceleration"] = -copy_preceding_df["xAcceleration"]
        copy_preceding_df["precedingXVelocity"] = -copy_preceding_df["precedingXVelocity"]
    copy_preceding_df["drivingDirection"] = car_item[car_item["id"] == p]["drivingDirection"].values[0]
    # 判断copy_preceding_df的xVelocity是否为正数, 若存在负数则把为负的数据设为0.01
    if not all(copy_preceding_df["xVelocity"] > 0):
        copy_preceding_df.loc[copy_preceding_df["xVelocity"] < 0, "xVelocity"] = 0.01
    return copy_preceding_df


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
        if name == 'preceding':
            sub_csv_path['frontCar'] = f'{folder_path}/{c}'
        elif name == 'following':
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
    return [f'./HighD_train_data/following_scene_withside_{following_time}/{s}' for s in sub_folders]


def process_drivingDirection(following_time):
    data_root = f'./HighD_train_data/following_scene_withside_{following_time}/'
    sub_folders = sub_folder(data_root, following_time)
    for s in tqdm(sub_folders):
        folder_name = s.split('/')[-1]
        new_folder = f'./HighD_train_data/following_scene_withside_changed_direction_{following_time}/{folder_name}'
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        csv_dict = sub_csv(s)
        for file_name, file_path in csv_dict.items():
            changed_df = change_direction(file_path)
            changed_df.to_csv(f'{new_folder}/{file_name}.csv', index=False)

