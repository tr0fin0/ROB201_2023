""" A set of robotics control functions """
import math
import numpy as np
import random


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # * TP1 Code
    distances = lidar.get_sensor_values()    # 
    angles = lidar.get_ray_angles()       # angles in radiums

    # angles and distances has n values where n is the resolution of the LIDAR

    for i in range(len(angles)):
        distance = distances[i]
        angle = angles[i]

        # print(f'({distance}, {angle})')
        if distance >= 40.0:
            # command is in [-1, +1], angle needs to be converted
            return {"forward": -distance/800, "rotation": random.uniform(-1, +1)}
        else:
            return {"forward": +distance/800, "rotation": random.uniform(-1, +1)}

    return {"forward": 0, "rotation": 0}


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """
    # * TP2
    distances = lidar.get_sensor_values()   # 
    angles = lidar.get_ray_angles()         # angles in radiums

    indexMin = np.argmin(distances)

    x = distances * np.cos(np.radians(angles))
    y = distances * np.sin(np.radians(angles))
    # poseLidar = [x, y, angles]
    # print(f'{poseLidar[0][0]} {poseLidar[1][0]} {poseLidar[2][0]}')
    # print(f'{distances[0]} {angles[0]}')
                  
    K_goal = 1
    dfx = K_goal * (poseLidar) / np.argmin(distances)

    command = {"forward": 0, "rotation": 0}

    return command
