"""
目的是为了生成测试集的列表
"""
import os
import random
import json
from tqdm import tqdm
def main():
    # 获取following_scene_withside文件夹下的所有文件名
    data_root = './HighD_train_data/following_scene_withside_30'
    sub_folders = os.listdir(data_root)
    test_json = {}
    for s in sub_folders:
        temp = s.split('_')[:4]
        tra_id = f'{temp[0]}_{temp[1]}_{temp[2]}'
        video_id = temp[0]
        preceding_id = temp[1]
        following_id = temp[2]
        lan_id = temp[3]
        test_id = {
            'video_id': video_id,
            'preceding_id': preceding_id,
            'following_id': following_id,
            'lan_id': lan_id
        }
        test_json.update({tra_id: test_id})
    with open(f"./HighD_train_data/test_list.json", "w", encoding="utf-8") as f:
        json.dump(test_json, f, ensure_ascii=False, indent=4)

    pass

if __name__ == '__main__':
    main()
