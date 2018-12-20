import cv2
import os
import numpy as np
import h5py
from util.World import World
from util.Vehicle import Vehicle
from util.Camera import Camera
from util.LaneMarking import LaneMarking
from util.Path import Path
import atexit

class Renderer:

    def __init__(self, debug = False):
        self.world = World()
        if not os.path.exists(self.world.save_path):
            raise ("No world available")
        self.world = self.world.load_world()
        self.camera = self.world.get_camera_from_actors()
        self.camera.set_transform(y = -1500)
        self.vehicle = Vehicle(self.camera, play =False)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)

        self.iter = 0
        self.recording = self.load_recording()
        Renderer.num_frames = len(self.recording)

        self.all_states = []
        self.debug = debug

        self.in_res = (72, 96)
        self.h5_file = h5py.File("data/pytorch_data.h5", "w")
        self.dset_data = self.h5_file.create_dataset("data", (0,3,self.in_res[0],self.in_res[1]), dtype=np.uint8,maxshape=(None, 3, self.in_res[0],self.in_res[1]), chunks=(1,3,self.in_res[0],self.in_res[1]))
        self.dset_labels = self.h5_file.create_dataset("labels", (0,1), dtype=np.float32, maxshape=(None, 1), chunks=(1,1))

        atexit.register(self.cleanup)

    def cleanup(self):
        self.h5_file.close()

    def get_sin_noise(self):
        noise = np.sin(self.iter / 10) # * 5 (increase amplitude)
        return noise

    def add_noise_over_camera(self):
        cam_params = list(self.camera.get_transform())
        noise = self.get_sin_noise()
        #cam_params[0] += noise * 10 #x
        #cam_params[2] += noise * 10 #z
        cam_params[4] += noise / 20 #yaw
        self.camera.set_transform(*cam_params)

    def load_recording(self):
        file = h5py.File("data/recording.h5", "r")
        recording = file["recording"][...]
        file.close()
        return recording

    def pre_simulate(self):
        for i in range(len(self.recording)):
            key = self.recording[self.iter, 0]
            mouse = (self.recording[self.iter, 1], self.recording[self.iter, 2])
            self.iter += 1
            self.vehicle.interpret_key(key)
            self.vehicle.interpret_mouse(mouse)
            self.vehicle.simulate()

            self.all_states.append([self.vehicle.T.copy(), self.camera.C.copy(), self.vehicle.next_locations.copy(), self.vehicle.vertices_W.copy(), self.vehicle.turn_angle])
        self.iter = 0
        self.path = Path(self.all_states)

    def render(self):
        channels = {}

        for i in range(len(self.all_states)):
            state = self.all_states[i]
            vehicle_T = state[0]
            camera_transform = state[1]
            vehicle_next_locations = state[2]
            vehicle_vertices_W = state[3]
            vehicle_turn_angle = state[4]

            self.camera.C = camera_transform
            self.vehicle.T = vehicle_T
            self.vehicle.next_locations = vehicle_next_locations
            self.vehicle.vertices_W = vehicle_vertices_W
            image_lanes = np.zeros((480, 640, 3), np.uint8)
            image_vehicle = np.zeros((480, 640, 3), np.uint8)
            image_path = np.zeros((480, 640, 3), np.uint8)

            for actor in self.world.actors:
                if type(actor) is Camera: continue
                if type(actor) is LaneMarking:
                    image_lanes = actor.render(image_lanes, self.camera)
            image_vehicle = self.vehicle.render(image_vehicle, self.camera)
            image_path = self.path.render(image_path, self.camera)
            #image_lanes = self.vehicle.render(image_lanes, self.camera)

            self.to_h5py([image_lanes,image_vehicle,image_path],vehicle_turn_angle,i)

            if self.debug:
                cv2.imshow("Image lanes", image_lanes)
                cv2.imshow("Image vehicle", image_vehicle)
                cv2.imshow("Image path", image_path)
                cv2.waitKey(1)

        print ("Rendering Done")

    def to_h5py(self, images = [], labels = (), index = -1):
        gray_images_resized = []
        for image in images:
            image_ = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_ = cv2.resize(image_,(self.in_res[1],self.in_res[0]))
            gray_images_resized.append(image_)
        image_concatenated = np.empty((3,self.in_res[0],self.in_res[1]), np.uint8)
        image_concatenated[0,...] = gray_images_resized[0]
        image_concatenated[1,...] = gray_images_resized[1]
        image_concatenated[2,...] = gray_images_resized[2]

        if self.debug:
            cv2.imshow("Image lanes", gray_images_resized[0])
            cv2.imshow("Image vehicle", gray_images_resized[1])
            cv2.imshow("Image path", gray_images_resized[2])
            cv2.waitKey(1)

        self.dset_data.resize((index + 1, 3, self.in_res[0],self.in_res[1]))
        self.dset_labels.resize((index + 1, 1))
        self.dset_data[index,...] = image_concatenated
        label_array = np.array([labels])
        self.dset_labels[index,...] = label_array


if __name__ =="__main__":

    # render = True
    # if render:
    #     renderer = Renderer(debug=False)
    #     renderer.pre_simulate()
    #     renderer.render()

    file = h5py.File("data/pytorch_data.h5", "r")
    dset_data = file['data']
    dset_labels = file['labels']
    for i in range(dset_data.shape[0]):
        image_bgr = dset_data[i,...]
        image_bgr = np.transpose(image_bgr,(1,2,0))
        print (image_bgr.shape)
        labels = dset_labels[i]
        cv2.imshow("image_bgr", image_bgr)
        cv2.waitKey(33)
    file.close()
