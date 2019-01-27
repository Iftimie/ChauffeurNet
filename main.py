from simulator.UI.WorldEditor import WorldEditor
from network.train import ConfigSimpleConv
from simulator.UI.TestNetwork import Simulator
from simulator.UI.Record import Recorder
from download_data import check_if_data_exists


#if errors about opencv DLL
#https://github.com/skvark/opencv-python/issues/154
#https://www.microsoft.com/en-us/software-download/mediafeaturepack

def main():
    check_if_data_exists()

    edit_world = True
    record = False
    do_train = False
    just_test_network = False

    if edit_world:
        worldEditor = WorldEditor("data/world.h5")
        worldEditor.run()

    if record:
        recorder = Recorder(event_bag_path="data/recorded_states.pkl", world_path="data/world.h5")
        recorder.run()

    if do_train:
        cfg = ConfigSimpleConv(root_path=".")
        for epoch in range(cfg.epochs):
            cfg.train(epoch)

    if just_test_network:
        simulator = Simulator(event_bag_path="data/recorded_states.pkl", network_path="data/ChauffeurNet.pt" ,
                              world_path="data/world.h5")
        simulator.run()


if __name__ =="__main__":
    main()