from simulator.util.Vehicle import Vehicle
from simulator.UI.Record import EventBag
from simulator.util.World import World
from simulator.util.Camera import Camera
from simulator.util.transform.util import params_from_tansformation
import pickle

def test_simulate_key():
    event_bag = EventBag("../../data/recording.h5", record=False)
    vehicle = Vehicle(Camera(), play=False)
    vehicle.set_transform(x=100)

    all_states = pickle.load(open("../../data/tmp_all_states.pkl", "rb"))
    # all_states = []
    # for i in range(len(event_bag)):
    #     key, x, y = event_bag.next_event()
    #     vehicle.simulate(key, (x, y))
    #
    #     all_states.append([vehicle.T.copy(), camera.C.copy(), vehicle.next_locations.copy(),
    #                             vehicle.vertices_W.copy(), vehicle.turn_angle])
    # pickle.dump(all_states, open("../../data/tmp_all_states.pkl", "wb"))
    # event_bag.reset()
    del event_bag
    return all_states


def test_simulate_waypoint(all_states):
    vehicle = Vehicle(Camera(), play=False)
    vehicle.set_transform(x=100)
    event_bag = EventBag("../../data/recording.h5", record=False)

    for state in all_states:
        key, x_mouse, y_mouse = event_bag.next_event()
        next_vehicle_T = state[0]
        x,y,z,roll,yaw,pitch = params_from_tansformation(next_vehicle_T)

        vehicle.interpret_key(key) # this is necessary for the speed
        vehicle.simulate_given_waypoint(x,z,yaw)

        x_n,y_n,z_n,roll_n,yaw_n,pitch_n = vehicle.get_transform()

        a=10


    del event_bag



if __name__ == "__main__":
    all_states = test_simulate_key()
    test_simulate_waypoint(all_states)
