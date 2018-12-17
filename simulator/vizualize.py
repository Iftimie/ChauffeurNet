import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np
import math
from util.World import World
from util.LaneMarking import LaneMarking
from util.Camera import Camera


def plot_world(vertices, K, R, T):
    plt.rcParams['figure.figsize'] = [6, 6]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    box_limit = 1000
    ax.set_xlim(-box_limit, box_limit)
    ax.set_ylim(-box_limit, box_limit)
    ax.set_zlim(-box_limit, box_limit)

    marker_size = 3
    ax.scatter(vertices[0, :], vertices[1, :], vertices[2, :], c="b", s=marker_size)

    # CAMERA_PART
    f = 250
    heading_vector = np.array([[0, 0, f]]).T
    heading_vector = R.dot(heading_vector)
    f_pixel_x = K[0, 0]
    f_pixel_y = K[1, 1]
    img_w = K[0, 2] * 2
    img_h = K[1, 2] * 2
    fov_x = 2 * math.atan(img_w / (2 * f_pixel_x)) * 180.0 / math.pi
    fov_y = 2 * math.atan(img_h / (2 * f_pixel_y)) * 180.0 / math.pi
    # oposite_len_x means the W/2 of the base of the pyramid
    # oposite_len_y means the H/2 of the base of the pyramid
    oposite_len_x = f * math.tan(math.radians(fov_x / 2))
    oposite_len_y = f * math.tan(math.radians(fov_y / 2))
    ox = oposite_len_x
    oy = oposite_len_y
    ax.quiver(T[0], T[1], T[2], heading_vector[0], heading_vector[1], heading_vector[2])
    ax.scatter(T[0], T[1], T[2], c="b")
    v = np.array([[0, 0, 0],  # v0 center
                  [ox, -oy, f],  # v1 right --------up
                  [ox, oy, f],  # v2 right----------down
                  [-ox, oy, f],  # v3 left---------down
                  [-ox, -oy, f]])  # v4 left--------up
    #           right face             bottom face        left face            upper face
    v = R.dot(v.T).T + T.T
    verts = [[v[0], v[1], v[2]], [v[0], v[2], v[3]], [v[0], v[3], v[4]], [v[0], v[4], v[1]]]
    # https://stackoverflow.com/questions/39408794/python-3d-pyramid
    trans = 0.2
    colour = (0.1, 0.8, 0.5, trans)
    ax.add_collection3d(Poly3DCollection(verts, facecolors=colour, linewidths=1, edgecolors='r', alpha=.15))

    plt.show()

cam_config = {"img_w":640, "img_h": 480, "f_cm":0.238, "pixel_width_cm":0.0003}
actors = []
actors.append(LaneMarking())
world = World(actors=actors)
camera = Camera(cam_config)
camera.set_transform(x = 0, y = -1000, z = 0, roll = 0, yaw = 0, pitch =-90)
plot_world(world.actors[0].vertices_W, camera.K, camera.T[:3,:3], camera.T[:3,3])