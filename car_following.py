from car_following_utils.following_scene_extraction import follow_scene_extraction
from car_following_utils.process_drivingDirection import process_drivingDirection
from car_following_utils.surround_aggregate import surround_aggregate


def main():
    car_following_time = 30
    # 首先提取车跟车场景
    follow_scene_extraction(car_following_time) 
    print("车跟车场景提取完成！")
    # 然后处理drivingDirection
    process_drivingDirection(car_following_time)
    print("drivingDirection处理完成！")
    # 最后进行左右车道车的聚合
    surround_aggregate(car_following_time)
    print("左右车道车聚合完成！")
    

if __name__ == "__main__":
    main()