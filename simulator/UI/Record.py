from simulator.util.Vehicle import Vehicle
from simulator.UI.GUI import GUI
import pickle
import atexit
from simulator.util.Camera import Camera
from config import Config

class Recorder(GUI):

    def __init__(self, event_bag_path="", world_path="" ):
        super(Recorder, self).__init__("Simulator", world_path=world_path)
        self.vehicle = Vehicle(self.camera)
        self.world.actors.append(self.vehicle)
        self.vehicle.is_active = True
        self.vehicle.render_next_locations_by_steering = True
        self.camera.is_active = False


        self.event_bag = EventBag(event_bag_path, record=True)

    # @Override
    def interpret_key(self):
        key = self.pressed_key
        if key == 27:
            self.running = False

    def run(self):
        while self.running:
            super(Recorder, self).interpretIO_and_render()

            self.world.simulate(self.pressed_key, GUI.mouse)
            to_save_dict = {}
            to_save_dict["pressed_key"] = self.pressed_key
            to_save_dict["mouse"] = (GUI.mouse[0], GUI.mouse[1])
            to_save_dict["vehicle"] = self.vehicle.get_relevant_states()

            self.event_bag.append(to_save_dict)

        print ("Game over")

import threading
import functools
import time
def synchronized(wrapped):
    lock = threading.Lock()
    # print lock, id(lock)
    @functools.wraps(wrapped)
    def _wrap(*args, **kwargs):
        with lock:
            print ("Calling '%s' with Lock %s from thread %s [%s]"
                   % (wrapped.__name__, id(lock),
                   threading.current_thread().name, time.time()))
            result = wrapped(*args, **kwargs)
            print ("Done '%s' with Lock %s from thread %s [%s]"
                   % (wrapped.__name__, id(lock),
                   threading.current_thread().name, time.time()))
            return result
    return _wrap

class EventBag:

    @synchronized
    def __init__(self, file_path, record = True):
        self.record = record
        if record == True:
            self.file = open( file_path, "wb" )
            self.list_states = []
        else:
            self.file = open( file_path, "rb" )
            self.list_states = pickle.load(self.file)
            self.file.close()
        self.crt_idx = 0
        atexit.register(self.cleanup)

    def append(self, events):
        if self.record == True:
            self.list_states.append(events)
            self.crt_idx +=1
        else:
            raise ValueError("EventBag opened as read mode")

    def __len__(self):
        return len(self.list_states)

    def next_event(self):
        if self.record == False:
            event = self.list_states[self.crt_idx]

            #update camera matrix, since recording could have been at another resolution
            camera = event["vehicle"]["camera"]
            camera.K = camera.create_K(Camera.cam_config)
            camera.C = camera.create_cammera_matrix(camera.T,camera.K)
            event["vehicle"]["camera"]= camera
            self.crt_idx +=1
        else:
            raise ValueError("EventBag opened as write mode")
        return event

    def __getitem__(self, idx):
        if self.record == False:
            event = self.list_states[idx]

            # update camera matrix, since recording could have been at another resolution
            camera = event["vehicle"]["camera"]
            camera.K = camera.create_K(Camera.cam_config)
            camera.C = camera.create_cammera_matrix(camera.T, camera.K)
            event["vehicle"]["camera"] = camera

        else:
            raise ValueError("EventBag opened as write mode")
        return event

    def reset(self):
        self.crt_idx = 0

    def cleanup(self):
        if self.record == True:
            pickle.dump(self.list_states, self.file)
            self.file.close()


if __name__ =="__main__":
    recorder = Recorder(event_bag_path="../../data/recorded_states.pkl", world_path="../../data/world.h5")
    recorder.run()
