# ChauffeurNet
Trying to implement (at least 10% hopefully, I just want the car to drive like 10 meters without crashing :worried: ) [ChauffeurNet](https://arxiv.org/pdf/1812.03079.pdf) : Learning to Drive by Imitating the Best and Synthesizing the Worst.

Development will be divided in the following steps:

1. Provide data generation tools:
  - [ ] ~~Add [Carla](https://github.com/carla-simulator/carla) as depenency. This will provide accurate rendered data.~~
  
  ![](assets/carla-sim.gif)
  - [x] Created my own simulator as I found carla to be too hard to use. This way I think I am more flexible as redering is done at train time based on recorded driving session.
  - [x] Provide preprocessing scripts for data and transform them into the required format for the network.
  
  
2. Implement some parts of the neural network:
  - [x] Implement steering in order to keep the center of the lane (Given predicted waypoints, compute the required turn angle to reach the waypoint, next is to compute the required speed)
  - [x] Implement path following
  - [ ] Implement speed control
  - [ ] Implement road mask layer
  - [ ] Implement agent box output layer
  - [x] Implement waypoint layer
  - [ ] Add other agents to input
  - [ ] Implement perception box output
  
3. Iterate from step 1 while adding more complexity

v 0.1 demo:

Basically, it is USELESS because the network only learned to predict waypoints along the desired path.
Given a waypoint the car computes the desired angle to reach that waypoint. No speed control is involved. Thus, I could just give to the car a point from the desired path.

The utility of predicted waypoints (of a complete implementation of ChauffeurNet) is that it takes into account other agents actions and driving rules, where hand crafted driving models would become too complex.

![](assets/first_net.gif)

How to run with pretrained model (will automatically download model from drive):

```bash
#For linux: sudo apt-get install python3-tk 
pip3 install <torch config ex: https://download.pytorch.org/whl/cpu/torch-1.0.0-cp36-cp36m-linux_x86_64.whl>
pip3 install -r requirements.txt
python3 main.py
```
