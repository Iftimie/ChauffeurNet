import numpy as np
from math import sin, cos, radians, degrees, pi, asin, atan2


def rot_y(deg=0):
    return np.array([[cos(deg), 0, sin(deg)],
                     [0, 1, 0],
                     [-sin(deg), 0, cos(deg)]])


def rot_z(deg=0):
    return np.array([[cos(deg), -sin(deg), 0],
                     [sin(deg), cos(deg), 0],
                     [0, 0, 1]])


def rot_x(deg=0):
    return np.array([[1, 0, 0],
                     [0, cos(deg), -sin(deg)],
                     [0, sin(deg), cos(deg)]])


def rotation_matrix(roll=0, yaw=0, pitch=0):
    """
    X axis positive to the right, negative to the right
    Y axis positive up, negative down
    Z axis positive forward, negative backwards
    :param roll:  angle in degrees around Z axis
    :param yaw:   angle in degrees around Y axis
    :param pitch: angle in degrees around X axis
    :return: 4x4 transformation matrix with rotation component. order is ZXY (roll, pitch, yaw).
    """
    roll = radians(roll)
    yaw = radians(yaw)
    pitch = radians(pitch)
    R = np.eye(4)
    R[:3, :3] = rot_y(yaw).dot(rot_x(pitch)).dot(rot_z(roll))
    return R


def translation_matrix(x=0, y=0, z=0):
    t = np.eye(4)
    t[0, 3] = x
    t[1, 3] = y
    t[2, 3] = z
    return t


def transformation_matrix(x=0, y=0, z=0, roll=0, yaw=0, pitch=0):
    """
    applies rotation then translation
    """
    R = rotation_matrix(roll, yaw, pitch)
    t = translation_matrix(x, y, z)
    T = t.dot(R)
    return T

def euler_angles(T):
    """
    :param T: works for a 4x4 matrix or a 3x3 matrix
    :return:
    """
    R = T[:3,:3]
    sx = -R[1][2]
    flt_eps = 0.00001
    if abs((abs(sx) - 1.0)) < flt_eps:
        if abs(sx - 1.0)  < flt_eps:
            thetaX = pi * 0.5
            thetaZ = 0
            thetaY = asin(-R[2][0])
        else:
            thetaX = -pi * 0.5
            thetaZ = 0
            thetaY = asin(-R[2][0])
    else:
        thetaX = asin(-R[1][2])
        thetaZ = atan2(R[1][0], R[1][1])
        thetaY = atan2(R[0][2], R[2][2])
    roll = degrees(thetaZ)
    yaw = degrees(thetaY)
    pitch = degrees(thetaX)
    return roll, yaw, pitch

def translation(T):
    """
    :param T: 4x4 matrix
    :return: x, y, z
    """
    return T[0,3], T[1,3], T[2, 3]

def params_from_tansformation(T):
    roll, yaw, pitch = euler_angles(T)
    x, y, z = translation(T)
    return x, y, z, roll, yaw, pitch