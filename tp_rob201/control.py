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
    atr_K = 1
    atr_diff = goal[:2] - pose[:2]
    atr_dist = np.sqrt(atr_diff[0]**2 + atr_diff[1]**2)

    if atr_dist > DST_LIM:
        atr_dF = atr_K * atr_diff / atr_dist

    elif atr_dist > DST_MIN:
        atr_dF = atr_K * atr_diff / DST_LIM

    else:
        command = {"forward": 0, "rotation": 0}
        return command

    atr_mag = np.sqrt(atr_dF[0]**2 + atr_dF[1]**2)
    atr_ang = (np.arctan2(atr_dF[1], atr_dF[0]) - pose[2]) / (2*np.pi)


    distances = lidar.get_sensor_values()   # 
    angles = lidar.get_ray_angles()         # angles in radiums

    # get array of bool's where the lidar found obstacules inside safe zone
    isDangerous = distances < DST_SAFE

    # create array of dangerous values
    dstDanger = distances[isDangerous]
    angDanger = angles[isDangerous]

    if len(dstDanger) != 0:
        # get the closest obstacule values that is dangerous

        minDst = np.min(dstDanger)
        minIdx = np.argmin(dstDanger)
        minAng = angDanger[minIdx]

        x_0 = pose[0]
        y_0 = pose[1]
        angle_0 = pose[2]

        obst = []
        obst.append(minDst * np.cos(minAng + angle_0) + x_0)
        obst.append(minDst * np.sin(minAng + angle_0) + y_0)
    else:
        print("errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
        obst = [0, 0]

    rep_K = 0.45e6
    rep_diff = obst[:2] - pose[:2]
    rep_dist = np.sqrt(rep_diff[0]**2 + rep_diff[1]**2)

    # if rep_dist > DST_LIM:
    if rep_dist <= DST_SAFE:
        rep_dF = rep_K / (rep_dist**3) * rep_diff * (1/rep_dist - 1/DST_SAFE)

    else:
        rep_dF = [0, 0]

    # rep_dF = np.linalg.norm(rep_dF)
    rep_mag = np.sqrt(rep_dF[0]**2 + rep_dF[1]**2)
    rep_ang = (np.arctan2(rep_dF[1], rep_dF[0]) - pose[2]) / (2*np.pi)

    dF = (atr_dF - rep_dF) / (atr_mag + rep_mag)
    mag = np.sqrt(dF[0]**2 + dF[1]**2)
    ang = (np.arctan2(dF[1], dF[0]) - pose[2]) / (2*np.pi)

    # mag = (atr_mag + )
    # forward = 0.25*atr_mag + 0.25*rep_mag
    # rotation = 0.85*atr_ang + 0.15*rep_ang

    forward = 0.30*mag
    rotation = 0.975*ang

    # print(" ")
    # print(f'{atr_dist:4.4f} {rep_dist:4.4f}')
    # print(f"{[f'{x:+4.4f}' for x in atr_dF]} {[f'{x:+4.4f}' for x in rep_dF]}")
    # print(f"{[f'{x:+4.4f}' for x in atr_diff]} {[f'{x:+4.4f}' for x in rep_diff]}")
    # print(f"{atr_mag:+4.4f} {atr_ang:+4.4f} {rep_mag:+4.4f} {rep_ang:+4.4f}")
    # print(" ")
    # print(f"{[f'{x:+4.4f}' for x in dF]}")
    # print(f"{forward:+4.4f} {rotation:+4.4f}")
    # print("___"*35)
    command = {"forward": forward, "rotation": rotation}

    return command
