import os
import pandas as pd
import numpy as np


def get_ego_feature(car_group_df, dict_cars):
    T = len(car_group_df)

    position = np.zeros((T, 2), dtype=np.float64)
    velocity = np.zeros((T, 2), dtype=np.float64)
    acceleration = np.zeros((T, 2), dtype=np.float64)
    shape = np.zeros((T, 2), dtype=np.float64)
    lane_id = np.zeros(T, dtype=np.int64)
    valid_mask = np.ones(T, dtype=np.bool_)

    if dict_cars[car_group_df["id"].values[0]]["class"]=='Car':
        car_class = np.array(0)
    elif dict_cars[car_group_df["id"].values[0]]["class"]=='Truck':
        car_class = np.array(1)
    else:
        raise TypeError("Unknown car class")
    drivingDirection = np.array(dict_cars[car_group_df["id"].values[0]]["drivingDirection"])
    for i in range(T):
        position[i, 0] = car_group_df["x"].values[i]
        position[i, 1] = car_group_df["y"].values[i]
        velocity[i, 0] = car_group_df["xVelocity"].values[i]
        velocity[i, 1] = car_group_df["yVelocity"].values[i]
        acceleration[i, 0] = car_group_df["xAcceleration"].values[i]
        acceleration[i, 1] = car_group_df["yAcceleration"].values[i]
        shape[i, 0] = car_group_df["width"].values[i]
        shape[i, 1] = car_group_df["height"].values[i]
        lane_id[i] = car_group_df["laneId"].values[i]

    return {
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "shape": shape,
        "lane_id": lane_id,
        "valid_mask": valid_mask,
        "class": car_class,
        "drivingDirection": drivingDirection,
    }


def main():
    MinT = 10 # 取最少存在10s的車輛才能作為ego car
    FrameRate = 25
    min_frame = MinT/(1/FrameRate) + 1  # 10s * 25fps = 250 frames

    for i in range(60):
        recordingMeta = f"./data/{i+1:02}_recordingMeta.csv"
        tracks = f"./data/{i+1:02}_tracks.csv"
        tracksMeta = f"./data/{i+1:02}_tracksMeta.csv"
        recordingMeta_df = pd.read_csv(recordingMeta)
        tracks_df = pd.read_csv(tracks)
        tracksMeta_df = pd.read_csv(tracksMeta)
        frameRate = recordingMeta_df["frameRate"].values[0]
        # 判断frameRate是否为25
        assert frameRate == FrameRate, "frameRate不为25"
        # 筛选出tracksMeta_df中numFrames中大于min_frame的行所对应的id
        valid_ids = tracksMeta_df[tracksMeta_df["numFrames"] >= min_frame]["id"].values
        dict_cars = {}
        for j in range(len(tracksMeta_df)):
            dict_cars[tracksMeta_df["id"].values[j]] = {
                "class": tracksMeta_df["class"].values[j],
                "width": tracksMeta_df["width"].values[j],
                "height": tracksMeta_df["height"].values[j],
                "numFrames": tracksMeta_df["numFrames"].values[j],
                "drivingDirection": tracksMeta_df["drivingDirection"].values[j],
            }
        for valid_id in valid_ids:
            car_df = tracks_df[tracks_df["id"] == valid_id]
            # 取car_df中的frame列
            car_frames = car_df["frame"].values
            # 判断car_frames是否连续,如果不连续，则分组，将连续的放一起
            frame_groups = np.split(car_frames, np.where(np.diff(car_frames) != 1)[0] + 1)
            for frame_group in frame_groups:
                if len(frame_group) < min_frame:
                    continue
                # 取car_df中car_df为frame_group的数据
                car_group_df = car_df[car_df["frame"].isin(frame_group)]
                ego_feature = get_ego_feature(car_group_df, dict_cars)

                print(1)



if __name__ == "__main__":
    main()
