from simulator.control.Controller import Controller

class TrafficLightsController(Controller):

    def __init__(self, traffic_light, world):
        super(TrafficLightsController, self).__init__(actor=traffic_light, world=world)
        self.counter = 0


    def step(self):

        pass