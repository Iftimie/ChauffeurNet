import cv2
import numpy as np

from util.World import World
from util.LaneMarking import LaneMarking
from util.Camera import Camera


cam_config = {"img_w":640, "img_h": 480, "f_cm":0.238, "pixel_width_cm":0.0003}
actors = []
actors.append(LaneMarking())
world = World(actors=actors)
camera = Camera(cam_config)
camera.set_transform(x = 0, y = 0, z = 0, roll = 0, yaw = 0, pitch = 0)


world.save_world()
world = world.load_world()

def mouse_listener(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pass
    elif event == cv2.EVENT_LBUTTONUP:
        pass
    print (event)

image = np.zeros((480,640), np.uint8)

cv2.namedWindow("Simulator")
cv2.setMouseCallback("Simulator", mouse_listener)

while True:
    image = world.render(image=image, C = camera)
    cv2.imshow("Simulator", image)

    key = cv2.waitKey(33)

