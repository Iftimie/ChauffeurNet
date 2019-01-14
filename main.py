from simulator.UI.WorldEditor import WorldEditor
from simulator.UI.DataGeneration import Renderer
from network.train import ConfigSimpleConv
from simulator.UI.TestNetwork import Simulator
from simulator.UI.Record import Recorder
from download_data import check_if_data_exists


def main():
    check_if_data_exists()

    edit_world = False
    record = True
    do_train = True
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