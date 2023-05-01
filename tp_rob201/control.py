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
    return reactiveRange(lidar, minClearance)
    # return reactiveWallFollow(lidar, minClearance)


def potential_field_control(lidar, pose, goal):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    pose : [x, y, theta] nparray, current pose in odom or world frame
    goal : [x, y, theta] nparray, target pose in odom or world frame
    """

    DST_SAFE = 175
    DST_LIM = 85
    DST_MIN = 50

    # * TP2
    # ! attractive potential
    att_const = 1
    att_difference = goal[:2] - pose[:2]
    atr_distance = np.sqrt(att_difference[0]**2 + att_difference[1]**2)

    potential_attractive = [0, 0]
    if atr_distance > DST_MIN:
        potential_attractive = att_const * att_difference / DST_LIM


    # ! repulsive potential
    # reading lidar data
    distances = lidar.get_sensor_values()   # 
    angles = lidar.get_ray_angles()         # angles in radiums

    # get array of bool's where the lidar found obstacules inside safe zone
    isDangerous = distances < DST_SAFE

    # create array of dangerous values
    danger_distances = distances[isDangerous]
    danger_angles    = angles[isDangerous]

    potential_repulsive = [0, 0]
    # checking for obstacules
    if len(danger_distances) != 0:
        # found, get the closest one
        min_index = np.argmin(danger_distances)

        min_distance = np.min(danger_distances)
        min_angle = danger_angles[min_index]

        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]

        obstacule = []
        obstacule.append(min_distance * np.cos(min_angle + angle_0) + x_0)
        obstacule.append(min_distance * np.sin(min_angle + angle_0) + y_0)

        repulsive_const = 0.9e6
        repulsive_diff = obstacule[:2] - pose[:2]
        repulsive_dist = np.sqrt(repulsive_diff[0]**2 + repulsive_diff[1]**2)

        if atr_distance > DST_MIN:
            if repulsive_dist <= DST_SAFE:
                potential_repulsive = repulsive_const / (repulsive_dist**3) * repulsive_diff * (1/repulsive_dist - 1/DST_SAFE)


    # ! resulting potential
    magnitude_attractive = np.sqrt(potential_attractive[0]**2 + potential_attractive[1]**2)
    magnitude_repulsive = np.sqrt(potential_repulsive[0]**2 + potential_repulsive[1]**2)

    if magnitude_attractive == 0 and magnitude_repulsive == 0:
        command = {"forward": 0, "rotation": 0}

    else:
        potential = (potential_attractive - potential_repulsive) / (magnitude_attractive + magnitude_repulsive)
        magnitude = np.sqrt(potential[0]**2 + potential[1]**2)
        angle = (np.arctan2(potential[1], potential[0]) - pose[2]) / (2*np.pi)

        forward  = 0.225 * magnitude
        rotation = 1.000 * angle

        command = {"forward": forward, "rotation": rotation}

    return command
