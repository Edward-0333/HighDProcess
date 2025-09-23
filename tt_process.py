import os
import pandas as pd
import numpy as np


def get_ego_feature(car_group_df, dict_cars, merged_num):
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
        "position": position[::int(merged_num)],
        "velocity": velocity[::int(merged_num)],
        "acceleration": acceleration[::int(merged_num)],
        "shape": shape[::int(merged_num)],
        "lane_id": lane_id[::int(merged_num)],
        "valid_mask": valid_mask[::int(merged_num)],
        "class": car_class,
        "drivingDirection": drivingDirection,
    }

def get_agent_feature(car_group_df, tracks_df, merged_num):
    ego_id = car_group_df["id"].values[0]
    frames = car_group_df["frame"].values[::int(merged_num)]
    # 因为frames过长，不像nuplan每一个场景为固定的时间长度，我们需要把过长的frames拆分成不同的场景，每个场景为10秒，采用移动窗口的方法。
    #
    valid_now = frames[25:len(frames)-101]
    valid_frames = [ frames[i-25:i+101] for i in range(25, len(frames)-101)]
    test = valid_frames[-1][25]
    for f in frames:
        # 获取tracks_df中frame为f的数据
        frame_f = tracks_df[tracks_df["frame"] == f]
        print(1)

def main():
    MinT = 10 # 取最少存在10s的車輛才能作為ego car
    FrameRate = 25
    merged_frame = 0.08 # 合并后的帧间隔

    # "加"是要保证冗余，加"2"是因为后续要取0.08(2帧合并成1帧)，252 = 2 + 250（250 = 25*2 + 100*2）
    min_frame = MinT/(1/FrameRate) + 2  # 10s * 25fps = 250 frames
    merged_num = merged_frame/(1/FrameRate)
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
                ego_feature = get_ego_feature(car_group_df, dict_cars, merged_num)
                get_agent_feature(car_group_df, tracks_df, merged_num)
                print(1)



if __name__ == "__main__":
    main()
