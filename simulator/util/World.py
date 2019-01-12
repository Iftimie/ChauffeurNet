from .Actor import Actor
from .Camera import Camera

#import util.Actor #Keep in mind that this is how you can import this package, among the other ways above
# print (sys.modules[__name__])
# print (dir(sys.modules[__name__]))
# print (sys.modules[__name__].__package__)
import sys
import os
import random
import string
import h5py
import numpy as np
import importlib


import threading
import functools
import time
def synchronized(wrapped):
    lock = threading.Lock()
    # print lock, id(lock)
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            # print ("Calling '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            result = wrapped(*args, **kwargs)
            # print ("Done '%s' with Lock %s from thread %s [%s]"
            #        % (wrapped.__name__, id(lock),
            #        threading.current_thread().name, time.time()))
            return result
    return _wrap

class World(Actor):

    def __init__(self, actors = [], world_path = "" ):
        super().__init__()
        self.actors = actors
        self.save_path = world_path
        pass

    #@Override
    def render(self, image = None, C = None, reset_image=True):
        if reset_image:
            image.fill(0)
        for actor in self.actors:
            image = actor.render(image, C)
        return image

    # @Override
    def simulate(self, pressed_key=None, mouse=None):
        for actor in self.actors:
            actor.simulate(pressed_key, mouse)

    def save_world(self, overwrite = False):
        for actor in self.actors:
            actor.set_inactive()
        directory = os.path.dirname(self.save_path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        if os.path.exists(self.save_path) and not overwrite:
            filename_ext = os.path.basename(self.save_path)
            filename, ext = os.path.splitext(filename_ext)
            UID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            filename = filename + UID +".h5"
            self.save_path = os.path.join(directory, filename)

        # The following spagetti code, takes the class names of all actors, counts the actors, and and creates a dataset for each type in h5py
        # No need to save the vehicle state (to_h5py), because in WorldEditor there is no vehicle
        dict_datasets = {} #{"name":[]}
        for actor in self.actors:
            if not actor.__class__.__name__ in dict_datasets.keys():
                dict_datasets[actor.__class__.__name__] = []
            actor_vect = actor.to_h5py()
            dict_datasets[actor.__class__.__name__].append(actor_vect)
        file = h5py.File(self.save_path, "w")
        print (dict_datasets.keys())
        print (len(dict_datasets["LaneMarking"]))
        print (len(dict_datasets["Camera"]))
        for class_name in dict_datasets.keys():

            list_actors_for_class_name = dict_datasets[class_name]
            all_actors = np.array(list_actors_for_class_name )
            dset = file.create_dataset(class_name, all_actors.shape,dtype=np.float32 )

            dset[...] = all_actors
        file.close()
        print ("world saved")

    def get_camera_from_actors(self):
        camera = None
        for actor in self.actors:
            if type(actor) is Camera:
                camera = actor
                camera.C = camera.create_cammera_matrix(camera.T,camera.K)
                # when camera is loaded from hdf5, the object of type camera is created, then only the T is initialized from hdf5, C remains uninitialized
        if camera is None:
            camera = Camera()
            self.actors.append(camera)
        return camera

    @synchronized
    def load_world(self):
        if not os.path.exists(self.save_path):
            raise ("No world available")
        file = h5py.File(self.save_path, "r")
        for class_name in file.keys():
            # module_imported =importlib.import_module("util")
            module_imported =importlib.import_module(sys.modules[__name__].__package__)
            class_ = getattr(module_imported, class_name)
            for i in range(file[class_name].shape[0]):
                instance = class_()

                instance.from_h5py(file[class_name][i])
                self.actors.append(instance)
        file.close()
        del file
