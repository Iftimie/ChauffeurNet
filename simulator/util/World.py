from .Actor import Actor
import _pickle as cPickle
import os
import random
import string
class World(Actor):

    def __init__(self, actors = [],):
        super().__init__()
        self.actors = actors
        pass

    #@Override
    def render(self, image = None, C = None):
        image.fill(0)
        for actor in self.actors:
            image = actor.render(image, C)
        return image

    def save_world(self, save_path = "data/world.pkl"):
        save_path
        directory = os.path.dirname(save_path)
        if not os.path.exists(directory):
            os.mkdir(directory)
        if os.path.exists(save_path):
            filename_ext = os.path.basename(save_path)
            filename, ext = os.path.splitext(filename_ext)
            UID = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
            filename = filename + UID +".pkl"
            save_path = os.path.join(directory, filename)
        fileObject = open(save_path, 'wb')
        cPickle.dump(self, fileObject, 2)
        fileObject.close()

    def load_world(self, save_path = "data/world.pkl"):
        fileObject = open(save_path, 'rb')
        world_object = cPickle.load(fileObject)
        fileObject.close()
        return world_object
