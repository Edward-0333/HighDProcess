import os
import pandas as pd
import numpy as np


def get_ego_feature(car_group_df, dict_cars, valid_now, valid_frames, merged_num):
    ego_features = []
    for v in valid_frames:
        T = len(v)

        position = np.zeros((T, 2), dtype=np.float64)
        velocity = np.zeros((T, 2), dtype=np.float64)
        acceleration = np.zeros((T, 2), dtype=np.float64)
        shape = np.zeros((T, 2), dtype=np.float64)
        lane_id = np.zeros(T, dtype=np.int64)
        valid_mask = np.ones(T, dtype=np.bool_)
        temp_car_group_df = car_group_df[car_group_df["frame"].isin(v)]
        drivingDirection = np.array(dict_cars[temp_car_group_df["id"].values[0]]["drivingDirection"])
        for i in range(T):
            position[i, 0] = temp_car_group_df["x"].values[i]
            position[i, 1] = temp_car_group_df["y"].values[i]
            velocity[i, 0] = temp_car_group_df["xVelocity"].values[i]
            velocity[i, 1] = temp_car_group_df["yVelocity"].values[i]
            acceleration[i, 0] = temp_car_group_df["xAcceleration"].values[i]
            acceleration[i, 1] = temp_car_group_df["yAcceleration"].values[i]
            shape[i, 0] = temp_car_group_df["width"].values[i]
            shape[i, 1] = temp_car_group_df["height"].values[i]
            lane_id[i] = temp_car_group_df["laneId"].values[i]
        ego_features.append({
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "shape": shape,
        "lane_id": lane_id,
        "valid_mask": valid_mask,
        "class": np.array(0),
        "drivingDirection": drivingDirection,
        })
    return ego_features

def get_agent_feature(dict_cars, car_group_df, tracks_df, valid_now, valid_frames, max_agents=50):
    ego_id = car_group_df["id"].values[0]
    agent_features = []
    for i, valid_frame in enumerate(valid_frames):
        print(i)
        now_i = valid_now[i]
        temp =tracks_df[tracks_df["frame"] == now_i]
        query_xy = temp[temp["id"] == ego_id][["x", "y"]].values[0]

        # 取tracks_df中frame为now_i的数据
        now_df = tracks_df[tracks_df["frame"] == now_i]
        # 删除id为ego_id的行
        now_df = now_df[now_df["id"] != ego_id]
        N,T = len(now_df), len(valid_frame)
        position = np.zeros((N, T, 2), dtype=np.float64)
        velocity = np.zeros((N, T, 2), dtype=np.float64)
        shape = np.zeros((N, T, 2), dtype=np.float64)
        acceleration = np.zeros((N, T, 2), dtype=np.float64)
        category = np.zeros((N,), dtype=np.int8)
        valid_mask = np.zeros((N, T), dtype=np.bool_)
        lane_id = np.zeros((N, T), dtype=np.int64)

        agent_ids = np.array(now_df['id'].values, dtype=np.int64)
        agent_cur_pos = np.array([now_df['x'].values, now_df['y'].values], dtype=np.float64).T
        distance = np.linalg.norm(agent_cur_pos - query_xy, axis=1)
        agent_ids_sorted = agent_ids[np.argsort(distance)[: max_agents]]
        agent_ids_dict = {agent_id: ii for ii, agent_id in enumerate(agent_ids_sorted)}
        for t, f in enumerate(valid_frame):
            temp = tracks_df[tracks_df["frame"] == f]
            tracked_objects_list = temp[temp["id"] != ego_id]
            for agent_id in tracked_objects_list['id'].values:
                if agent_id not in agent_ids_dict:
                    continue
                idx = agent_ids_dict[agent_id]
                position[idx, t] = tracked_objects_list[tracked_objects_list["id"] == agent_id][["x", "y"]].values[0]
                velocity[idx, t] = tracked_objects_list[tracked_objects_list["id"] == agent_id][["xVelocity", "yVelocity"]].values[0]
                acceleration[idx, t] = tracked_objects_list[tracked_objects_list["id"] == agent_id][["xAcceleration", "yAcceleration"]].values[0]
                shape[idx, t] = tracked_objects_list[tracked_objects_list["id"] == agent_id][["width", "height"]].values[0]
                lane_id[idx, t] = tracked_objects_list[tracked_objects_list["id"] == agent_id]["laneId"].values[0]
                valid_mask[idx, t] = True
                if f == now_i:
                    category[idx] = 1
        agent_feature = {
            "position": position,
            "velocity": velocity,
            "acceleration": acceleration,
            "shape": shape,
            "category": category,
            "valid_mask": valid_mask,
            "lane_id": lane_id,
        }
        agent_features.append(agent_feature)
    return agent_features

def get_ego_current_state(car_group_df, valid_now):
    current_states = []
    for v in valid_now:
        ego_state = np.zeros(5,dtype=np.float64)
        temp = car_group_df[car_group_df["frame"] == v]
        ego_state[0:2] = temp[["x", "y"]].values[0]
        ego_state[2:4] = temp[["xVelocity", "yVelocity"]].values[0]
        ego_state[4] = temp["xAcceleration"].values[0]
        current_states.append(ego_state)
    return current_states


def move_windows(car_group_df,merged_num):
    ego_id = car_group_df["id"].values[0]
    frames = car_group_df["frame"].values[::int(merged_num)]
    # 因为frames过长，不像nuplan每一个场景为固定的时间长度，我们需要把过长的frames拆分成不同的场景，每个场景为10秒，采用移动窗口的方法。
    valid_now = frames[25:len(frames) - 100]
    valid_frames = [frames[i - 25:i + 101] for i in range(25, len(frames) - 100)]
    return valid_now, valid_frames

def main():
    MinT = 10 # 取最少存在10s的車輛才能作為ego car
    FrameRate = 25
    merged_frame = 0.08 # 合并后的帧间隔

    # "加"是要保证冗余，加"2"是因为后续要取0.08(2帧合并成1帧)，252 = 2 + 250（250 = 25*2 + 100*2）
    min_frame = MinT/(1/FrameRate) + 2  # 10s * 25fps = 250 frames
    merged_num = merged_frame/(1/FrameRate)
    for i in range(60):
        print('正在处理第',i+1,'个场景')
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
                valid_now, valid_frames = move_windows(car_group_df,merged_num)
                current_states = get_ego_current_state(car_group_df, valid_now)
                ego_feature = get_ego_feature(car_group_df, dict_cars, valid_now, valid_frames, merged_num)
                agent_feature = get_agent_feature(dict_cars, car_group_df, tracks_df, valid_now, valid_frames)
                # 保存数据
                for jj in range(len(valid_now)):
                    save_path = f"./processed_data/recording_{i+1:02}/scene_{valid_id}_{jj}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    np.savez_compressed(os.path.join(save_path, "ego_feature.npz"), **ego_feature[jj])
                    np.savez_compressed(os.path.join(save_path, "agent_feature.npz"), **agent_feature[jj])
                    np.savez_compressed(os.path.join(save_path, "current_state.npz"), current_states[jj])

                print(1)



if __name__ == "__main__":
    main()
