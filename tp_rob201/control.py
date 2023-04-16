""" A set of robotics control functions """
import math
import numpy as np
import random


DST_SAFE = 175
DST_LIM = 100
DST_MIN = 25

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
    frontIndex = 180
    rangeSize = 60

    start = int(frontIndex-rangeSize/2)
    end = int(frontIndex+rangeSize/2)

    if distFOV(lidar, start, end) >= minClearance:
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

    minClearance = 50.0
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

    K = 1
    diff = goal[:2] - pose[:2]
    dist = np.sqrt(diff[0]**2 + diff[1]**2)

    if dist > DIST_MIN:
        dF = K * diff / dist

    elif dist > ROBOT_SIZE:
        dF = K * diff / DIST_MIN

    else:
        return {"forward": 0, "rotation": 0}

    mag = np.sqrt(dF[0]**2 + dF[1]**2)
    ang = (np.arctan2(dF[1], dF[0]) - pose[2]) / (2*np.pi)

    command = {"forward": 0.25*mag, "rotation": ang}

    return command
