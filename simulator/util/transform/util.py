import numpy as np
from math import sin, cos, radians, degrees


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
    :return: 4x4 transformation matrix with rotation component. order is XYZ (pitch, yaw, roll).
    """
    roll = radians(roll)
    yaw = radians(yaw)
    pitch = radians(pitch)
    R = np.eye(4)
    R[:3, :3] = rot_z(roll).dot(rot_y(yaw).dot(rot_x(pitch)))
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
