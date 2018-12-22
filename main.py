from simulator.UI.WorldEditor import WorldEditor
from simulator.UI.DataGeneration import Renderer
from network.train import ConfigSimpleConv
from simulator.UI.TestNetwork import Simulator
from simulator.UI.Record import Recorder
from download_data import check_if_data_exists


def main():
    check_if_data_exists()
    worldEditor = WorldEditor("data/world.h5")
    worldEditor.run()
    # recorder = Recorder(event_bag_path="data/recording.h5", world_path="data/world.h5")
    # recorder.run()
    # renderer = Renderer(world_path="data/world.h5", h5_path="data/pytorch_data.h5",
    #                     event_bag_path="data/recording.h5")
    # renderer.render()
    # renderer.visualize()

    # cfg = ConfigSimpleConv()
    # for epoch in range(cfg.epochs):
    #     cfg.train(epoch)
    #     # test(args, model, device, test_loader)
    # pass
    simulator = Simulator(event_bag_path="data/recording.h5", network_path="data/ChauffeurNet.pt" ,
                          world_path="data/world.h5")
    simulator.run()


if __name__ =="__main__":
    main()