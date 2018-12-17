from .Actor import Actor

class World(Actor):

    def __init__(self, actors = []):
        super().__init__()
        self.actors = actors
        pass

    #@Override
    def render(self, image = None, C = None):
        image.fill(0)
        for actor in self.actors:
            image = actor.render(image, C)
        return image
