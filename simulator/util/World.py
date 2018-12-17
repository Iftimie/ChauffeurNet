from .Actor import Actor
from .Camera import Camera
import _pickle as cPickle
import os
import random
import string
class World(Actor):

    def __init__(self, actors = [],):
        super().__init__()
        self.actors = actors
        self.save_path = "data/world.pkl"
        pass

    #@Override
    def render(self, image = None, C = None):
        image.fill(0)
        for actor in self.actors:
            image = actor.render(image, C)
        return image

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
            filename = filename + UID +".pkl"
            save_path = os.path.join(directory, filename)
        fileObject = open(self.save_path, 'wb')
        cPickle.dump(self, fileObject, 2)
        fileObject.close()
        print ("world saved")

    def get_camera_from_actors(self):
        camera = None
        for actor in self.actors:
            if type(actor) is Camera:
                camera = actor
        if camera is None:
            camera = Camera()
            self.actors.append(camera)
        return camera

    def load_world(self):
        fileObject = open(self.save_path, 'rb')
        world_object = cPickle.load(fileObject)
        fileObject.close()
        return world_object
