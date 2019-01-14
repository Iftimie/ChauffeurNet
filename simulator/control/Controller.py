
class Controller:

    def __init__(self, actor, world):
        """
        :param actor:
        """
        self.registered_actor = actor
        self.world = world

    def step(self):
        # Apply operations on self.registered_actor
        pass