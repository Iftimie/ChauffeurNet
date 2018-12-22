import os
from simulator.util.Vehicle import Vehicle
from simulator.UI.GUI import GUI
from simulator.UI.GUI import EventBag

class Recorder(GUI):

    def __init__(self, event_bag_path="", world_path="" ):
        super(Recorder, self).__init__("Simulator", world_path=world_path)
        self.camera.set_transform(y=-1500)
        self.vehicle = Vehicle(self.camera)
        self.vehicle.set_transform(x = 100)
        self.world.actors.append(self.vehicle)
        self.vehicle.is_active = True
        self.camera.is_active = False

        self.event_bag = EventBag(event_bag_path, record=True)

    # @Override
    def interpret_key(self):
        key = self.pressed_key
        if key == 27:
            self.running = False

    def run(self):

        while self.running:
            self.interact()
            self.world.simulate(self.pressed_key, GUI.mouse)
            self.event_bag.append([self.pressed_key, GUI.mouse[0], GUI.mouse[1]])

        print ("Game over")


if __name__ =="__main__":
    recorder = Recorder(event_bag_path="../../data/recording.h5", world_path="../../data/world.h5")
    recorder.run()
