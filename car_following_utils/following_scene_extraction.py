import pandas as pd
import numpy as np
import time
import os


def filter_time_list(filter_df_list, min_frames):
    new_filter_df_list = []

    for df in filter_df_list:
        if df.shape[0] < min_frames:
            continue
        assert df.shape[0] < min_frames * 8, "行数大于min_frames * 8"

        if df.shape[0] >= min_frames * 2 and df.shape[0] < min_frames * 3:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        elif df.shape[0] >= min_frames * 3 and df.shape[0] < min_frames * 4:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[min_frames:2*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        elif df.shape[0] >= min_frames * 4 and df.shape[0] < min_frames * 5:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[min_frames:2*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[2*min_frames:3*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        elif df.shape[0] >= min_frames * 5 <= min_frames * 6:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[min_frames:2*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[2*min_frames:3*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[3*min_frames:4*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        elif df.shape[0] >= min_frames * 6 <= min_frames * 7:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[min_frames:2*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[2*min_frames:3*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[3*min_frames:4*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[4*min_frames:5*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        elif df.shape[0] >= min_frames * 7 <= min_frames * 8:
            # 将df的前min_frames行数据保存到new_df中
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[min_frames:2*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[2*min_frames:3*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[3*min_frames:4*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[4*min_frames:5*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的中间min_frames行数据保存到new_df中
            new_df = df.iloc[5*min_frames:6*min_frames]
            new_filter_df_list.append(new_df)
            # 将df的后min_frames行数据保存到new_df中
            new_df = df.iloc[-min_frames:]
            new_filter_df_list.append(new_df)
        else:
            new_df = df.iloc[:min_frames]
            new_filter_df_list.append(new_df)

    return new_filter_df_list


def drop_change_lane(car_df, drop_seconds=3):
    drop_frame = round(drop_seconds/0.04)
    # 获取car_df的laneId并去重
    laneId = car_df["laneId"].values
    laneId = list(set(laneId))
    copy_car_df = car_df.copy().reset_index()
    following_dict = {}
    # 判断是否存在换道行为
    if len(laneId) > 1:  # 存在换道行为，需要将换道时刻前后3s的数据去除
        for l_id in laneId:
            lane_df = car_df[car_df["laneId"] == l_id]
            start_frame = lane_df["frame"].values[0]  # 获取lane_df的第一行的frame
            end_frame = lane_df["frame"].values[-1]  # 获取lane_df的最后一行的frame

            # 判断start_frame在car_df中第几行
            start_index = copy_car_df[copy_car_df["frame"] == start_frame].index[0]
            if start_index != 0 :
                if lane_df.shape[0] > drop_frame:
                    lane_df = lane_df.iloc[drop_frame:]
                    following_dict[l_id] = lane_df
                else:
                    continue
            # 判断end_frame在car_df中是否是最后一行
            end_index = copy_car_df[copy_car_df["frame"] == end_frame].index[0]
            if end_index != copy_car_df.shape[0] - 1:
                if lane_df.shape[0] > drop_frame:
                    lane_df = lane_df.iloc[:-drop_frame]
                    following_dict[l_id] = lane_df
                else:
                    continue
    else:
        following_dict[laneId[0]] = car_df

    return following_dict



def extract_left_right(tracks_df, following_df, preceding_df, tracksMeta_df):
    # 获取following_df的laneId
    following_laneId = following_df["laneId"].values[0]
    # 获取preceding_df的drivingDirection
    following_drivingDirection = following_df["drivingDirection"].values[0]
    if following_drivingDirection ==  1:
        left_lane_id = following_laneId + 1 # left_Lane_ID
        right_lane_id = following_laneId - 1 # right_Lane_ID
    elif following_drivingDirection == 2:
        left_lane_id = following_laneId - 1 # left_Lane_ID
        right_lane_id = following_laneId + 1 # right_Lane_ID

    frames = following_df["frame"].values # 获取following_df的frame
    same_time_df = tracks_df[tracks_df["frame"].isin(frames)] # 获取tracks_df中frame为与frames相同的数据
    same_time_df = same_time_df.drop_duplicates()  # same_time_df去重

    # 获取same_time_df中的Lane_ID为left_lane_id的数据
    same_time_left_df = same_time_df[same_time_df['laneId'] == left_lane_id]
    # 获取same_time_df中的Lane_ID为right_lane_id的数据
    same_time_right_df = same_time_df[same_time_df['laneId'] == right_lane_id]
    if len(same_time_left_df) > 0:
        # 获取same_time_left_df的第一行的id
        temp_id = same_time_left_df["id"].values[0]
        # 获取tracksMeta_df中id为temp_id的drivingDirection
        left_drivingDirection = tracksMeta_df[tracksMeta_df["id"] == temp_id]["drivingDirection"].values[0]
        if left_drivingDirection != following_drivingDirection:
            same_time_left_df = pd.DataFrame(columns=following_df.columns)
    if len(same_time_right_df) > 0:
        # 获取same_time_right_df的第一行的id
        temp_id = same_time_right_df["id"].values[0]
        # 获取tracksMeta_df中id为temp_id的drivingDirection
        right_drivingDirection = tracksMeta_df[tracksMeta_df["id"] == temp_id]["drivingDirection"].values[0]
        if right_drivingDirection != following_drivingDirection:
            same_time_right_df = pd.DataFrame(columns=following_df.columns)

    following_width = following_df["width"].values[0]  # 获取copy_following_df的width
    preceding_width = preceding_df["width"].values[0]  # 获取copy_preceding_df的width

    # 创建与copy_following_df相同列的DataFrame
    left_preceding_df = pd.DataFrame(columns=following_df.columns)
    left_alongside_df = pd.DataFrame(columns=following_df.columns)
    right_preceding_df = pd.DataFrame(columns=following_df.columns)
    right_alongside_df = pd.DataFrame(columns=following_df.columns)

    if len(same_time_left_df) == 0 and len(same_time_right_df) == 0:
        return left_preceding_df, left_alongside_df, right_preceding_df, right_alongside_df

    for f in frames:
        # 获取copy_following_df中frame为f的x
        following_x = following_df[following_df["frame"] == f]["x"].values[0]
        # 获取copy_preceding_df中frame为f的x
        preceding_x = preceding_df[preceding_df["frame"] == f]["x"].values[0]
        dhw = abs(preceding_x - following_x) - preceding_width

        # 获取same_time_left_df中frame为f的数据
        left_t_df = same_time_left_df[same_time_left_df["frame"] == f]
        # 获取same_time_right_df中frame为f的数据
        right_t_df = same_time_right_df[same_time_right_df["frame"] == f]

        if len(left_t_df)> 0:
            if following_drivingDirection == 1:
                dhl =  following_x - left_t_df['x'] - left_t_df['width']
            elif following_drivingDirection == 2:
                dhl = left_t_df['x'] - following_x - following_width
            # dhl = abs(left_t_df['x']- following_x) - left_t_df['width']
            left_t_df_copy = left_t_df.copy()
            left_t_df_copy['dhl'] = dhl
            # 获取left_t_df中dhl < dhw的数据并且dhl > v_length的数据
            new_line = left_t_df_copy[(left_t_df_copy['dhl'] < dhw) & (left_t_df_copy['dhl'] > 0)]
            # 取new_line中dhl最小的数据
            new_line = new_line[new_line['dhl'] == new_line['dhl'].min()]
            left_preceding_df = pd.concat([left_preceding_df, new_line], ignore_index=True)
            # 获取left_t_df中dhl < v_length的数据并且dhl > -following_car_length的数据
            new_line = left_t_df_copy[(left_t_df_copy['dhl'] < 0) & (left_t_df_copy['dhl'] > -following_width)]

            left_alongside_df = pd.concat([left_alongside_df, new_line], ignore_index=True)

        if len(right_t_df) > 0 :
            if following_drivingDirection == 1:
                dhr =  following_x - right_t_df['x'] - left_t_df['width']
            elif following_drivingDirection == 2:
                dhr = right_t_df['x'] - following_x - following_width
            # dhr =  abs(right_t_df['x'] - following_x) - right_t_df['width']
            right_t_df_copy = right_t_df.copy()
            right_t_df_copy['dhr'] = dhr
            # 获取right_t_df中dhr < dhw的数据并且dhr > v_length的数据
            new_line = right_t_df_copy[(right_t_df_copy['dhr'] < dhw) & (right_t_df_copy['dhr'] > 0)]
            # 取new_line中dhr最小的数据
            new_line = new_line[new_line['dhr'] == new_line['dhr'].min()]
            right_preceding_df = pd.concat([right_preceding_df, new_line], ignore_index=True)
            # 获取right_t_df中dhr < v_length的数据并且dhr > -following_car_length的数据
            new_line = right_t_df_copy[(right_t_df_copy['dhr'] < 0) & (right_t_df_copy['dhr'] > - following_width)]
            right_alongside_df = pd.concat([right_alongside_df, new_line], ignore_index=True)
        left_preceding_df['drivingDirection'] = following_drivingDirection
        left_alongside_df['drivingDirection'] = following_drivingDirection
        right_preceding_df['drivingDirection'] = following_drivingDirection
        right_alongside_df['drivingDirection'] = following_drivingDirection
    return left_preceding_df, left_alongside_df, right_preceding_df, right_alongside_df


def extract_proceeding_data(tracks_df, following_df, p, tracksMeta_df):

    # 获取tracks_df中id为p的车辆
    preceding_df = tracks_df[tracks_df["id"] == p]
    # 筛选出preceding_df中与car_df相同frame的数据
    preceding_df = preceding_df[preceding_df["frame"].isin(following_df["frame"].values)]
    # 判断preceding_df的行数是否与car_df的行数相等
    assert preceding_df.shape[0] == following_df.shape[0], "行数不相等"
    # 获取tracksMeta_df中id为p的drivingDirection
    preceding_drivingDirection = tracksMeta_df[tracksMeta_df["id"] == p]["drivingDirection"].values[0]
    # 使preceding_df中新增一列drivingDirection,并赋值为preceding_drivingDirection
    preceding_df["drivingDirection"] = preceding_drivingDirection

    return preceding_df


def filter_100_distance(following_df):
    # return [following_df]  # 不筛选距离了
    min_100_list = []
    # 筛选出dwh小于100的数据
    following_df_100 = following_df[following_df["dhw"] < 100]
    if following_df.shape[0] == following_df_100.shape[0]:
        return [following_df_100]
    else:
        if following_df_100.shape[0] == 0:
            return min_100_list
        # # 去除调min_100_list中的第50行数据(调试用)
        # following_df_100 = following_df_100.drop(following_df_100.index[50])
        # 根据following_df_100的frame的连贯性进行筛选
        now_i = 0
        for i in range(following_df_100.shape[0]-1):
            data_i_frame = following_df_100.iloc[i]['frame']
            data_ii_frame = following_df_100.iloc[i + 1]['frame']
            if data_ii_frame - data_i_frame == 1:
                continue
            else:
                new_df = following_df_100.iloc[now_i:i + 1]
                min_100_list.append(new_df)
                now_i = i + 1
        new_df = following_df_100.iloc[now_i:i + 1]
        min_100_list.append(new_df)

    return min_100_list


def extract_following_data(following_lane, p, tracksMeta_df, min_frames):
    # 获取following_lane中precedingId为p的数据
    following_df = following_lane[following_lane["precedingId"] == p]

    # 获取following_df中的id
    followingId = following_df["id"].values[0]
    # 获取tracksMeta_df中id为followingId的drivingDirection
    following_drivingDirection = tracksMeta_df[tracksMeta_df["id"] == followingId]["drivingDirection"].values[0]
    # 获取tracksMeta_df中id为p的drivingDirection
    preceding_drivingDirection = tracksMeta_df[tracksMeta_df["id"] == p]["drivingDirection"].values[0]
    assert following_drivingDirection == preceding_drivingDirection, "drivingDirection不相等"
    copy_following_df = following_df.copy()
    # 使following_df中新增一列drivingDirection,并赋值为following_drivingDirection
    copy_following_df["drivingDirection"] = following_drivingDirection
    filter_df_list = filter_100_distance(copy_following_df)
    filter_df_list = filter_time_list(filter_df_list, min_frames)
    if len(filter_df_list) > 1:
        print(1)
    return filter_df_list



def follow_scene_extraction(following_time, drop_seconds=3):
    """
    following_time: seconds, 需要提取的跟随时间
    drop_seconds: seconds, 换道前后需要去除的时间
    """
    for i in range(60):
        recordingMeta = f"./data/{i+1:02}_recordingMeta.csv"
        tracks = f"./data/{i+1:02}_tracks.csv"
        tracksMeta = f"./data/{i+1:02}_tracksMeta.csv"
        recordingMeta_df = pd.read_csv(recordingMeta)
        tracks_df = pd.read_csv(tracks)
        tracksMeta_df = pd.read_csv(tracksMeta)
        frameRate = recordingMeta_df["frameRate"].values[0]
        # 判断frameRate是否为25
        assert frameRate == 25, "frameRate不为25"
        recording_time = 1/frameRate
        min_frames = int(following_time/recording_time)
        print('min_frames:', min_frames)
        time.sleep(3)

        # 筛选出出现时间大于15s的车辆
        car_item = tracksMeta_df[tracksMeta_df["numFrames"] > min_frames]
        # 筛选出class仅为Car的车辆
        car_item = car_item[car_item["class"] == "Car"]
        # 获取所有车辆id
        car_class_id = car_item["id"].values
        for car in car_class_id:
            print(f'正在处理第{i+1:02}个视频的第{car}辆车')
            # 在tracks_df中筛选出id为car的车辆
            car_df = tracks_df[tracks_df["id"] == car]
            # 验证car_tracks_df的行数是否于car_df中的numFrames相等
            assert car_df.shape[0] == car_item[car_item["id"] == car]["numFrames"].values[0], "行数不相等"
            following_dict = drop_change_lane(car_df, drop_seconds=drop_seconds)  # 去除换道行为，返回每个车在单独车道上的数据
            for lane_id, following_lane in following_dict.items():
                # 获取following_df的precedingId并去重
                precedingId = following_lane["precedingId"].values
                precedingId = list(set(precedingId))
                for p in precedingId:
                    if p and p in car_class_id:
                        following_df_list = extract_following_data(following_lane, p, tracksMeta_df, min_frames)
                        for n, following_df in enumerate(following_df_list):
                            # if following_df.shape[0] < min_frames:
                            #     continue
                            preceding_df = extract_proceeding_data(tracks_df, following_df, p, tracksMeta_df)
                            left_preceding_df, left_alongside_df, right_preceding_df, right_alongside_df = (
                                extract_left_right(tracks_df, following_df, preceding_df, tracksMeta_df))
                            save_dir = f"./HighD_train_data/following_scene_withside_{following_time}/{i + 1:02}_{p}_{car}_l{lane_id}_n{n}"
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)


                            assert preceding_df.shape[0] == min_frames, "行数有问题"
                            assert following_df.shape[0] == min_frames, "行数有问题"
                            preceding_df.to_csv(f"{save_dir}/preceding.csv", index=False)
                            following_df.to_csv(f"{save_dir}/following.csv", index=False)
                            if left_preceding_df.shape[0] > 0:
                                left_preceding_df.to_csv(f'{save_dir}/left_preceding.csv', index=False)
                            if left_alongside_df.shape[0] > 0:
                                left_alongside_df.to_csv(f'{save_dir}/left_alongside.csv', index=False)
                            if right_preceding_df.shape[0] > 0:
                                right_preceding_df.to_csv(f'{save_dir}/right_preceding.csv', index=False)
                            if right_alongside_df.shape[0] > 0:
                                right_alongside_df.to_csv(f'{save_dir}/right_alongside.csv', index=False)



if __name__ == "__main__":
    follow_scene_extraction(following_time=30)
