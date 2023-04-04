""" A set of robotics control functions """
import math
import numpy as np
import random


def reactiveFront(lidar, minClearance: float):
    distances = lidar.get_sensor_values()   # distance in cm
    frontIndex = 180


    if distances[frontIndex] >= minClearance:
        return {"forward": +0.25, "rotation": 0.00}
    else:
        return {"forward": +0.00, "rotation": random.uniform(-1, +1)}


def distFOV(lidar, start: int, end: int):
    return np.mean(lidar.get_sensor_values()[start:end])


def reactiveRange(lidar, minClearance: float):
    distances = lidar.get_sensor_values()   # distance in cm
    frontIndex = 180
    rangeSize = 30

    start = int(frontIndex-rangeSize/2)
    end = int(frontIndex+rangeSize/2)

    rangeValue = np.mean(distances[start:end])

    if rangeValue >= minClearance:
        return {"forward": +0.25, "rotation": 0.00}
    else:
        return {"forward": +0.00, "rotation": random.uniform(-1, +1)}


def reactiveWallFollow(lidar, targetClearance: float):
    distances = lidar.get_sensor_values()   # distance in cm
    distB = distances[0]    # distance Back
    distR = distances[90]   # distance Right
    # distF = distances[180]  # distance Front
    distF = distFOV(lidar, 170, 190)  # distance Front
    # distL = distances[270]  # distance Left
    distL = distFOV(lidar, 260, 280)  # distance Left


    
    # print(f'\t\t{distF:4.0f}')
    # print(f'\t{distL:4.0f}\t\t{distR:4.0f}')
    # print(f'\t\t{distB:4.0f}')
    print(f'{distL:4.4f} {distF:4.4f}')

    if distF >= 100:
        return {"forward": +0.125, "rotation": 0.0}

    else:
        if distL >= 60:
            return {"forward": +0.125, "rotation": -0.5}

        else:
            error = (distL - targetClearance) / distL

            angle = +0.5 * error
            # print(f'a: {angle} e: {error}')

            return {"forward": +0.125, "rotation": angle}


def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # * TP1 Code

    minClearance = 40.0
    # return reactiveFront(lidar, minClearance)
    # return reactiveRange(lidar, minClearance)
    return reactiveWallFollow(lidar, minClearance)


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
