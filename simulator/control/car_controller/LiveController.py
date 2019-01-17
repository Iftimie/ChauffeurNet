from simulator.control.Controller import Controller

class LiveController(Controller):
    """
    This controller is used for controlling the live
    """

    def __init__(self, vehicle, world):
        super(LiveController, self).__init__(actor=vehicle, world=world)


    def step(self, pressed_key, mouse):
        self.registered_actor.simulate(pressed_key, mouse)

