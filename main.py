from simulator.UI.WorldEditor import WorldEditor
from simulator.UI.DataGeneration import Renderer
from network.train import ConfigSimpleConv
from simulator.UI.TestNetwork import Simulator

def main():
    # worldEditor = WorldEditor("simulator/data/world.h5")
    # worldEditor.run()
    # renderer = Renderer(world_path="simulator/data/world.h5", h5_path="simulator/data/pytorch_data.h5",
    #                     event_bag_path="simulator/data/recording.h5")
    # renderer.render()
    # renderer.visualize()

    # cfg = ConfigSimpleConv()
    # for epoch in range(cfg.epochs):
    #     cfg.train(epoch)
    #     # test(args, model, device, test_loader)
    # pass
    simulator = Simulator(event_bag_path="simulator/data/recording.h5", network_path="network/ChauffeurNet.pt" ,
                          world_path="simulator/data/world.h5")
    simulator.run()


if __name__ =="__main__":
    main()