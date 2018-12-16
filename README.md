# ChauffeurNet
Trying to implement (at least 10% hopefully, I just want the car to drive like 10 meters without crashing :worried: ) [ChauffeurNet](https://arxiv.org/pdf/1812.03079.pdf) : Learning to Drive by Imitating the Best and Synthesizing the Worst.

Development will be divided in the following steps:

1. Provide data generation tools:
  - [ ] Add [Carla](https://github.com/carla-simulator/carla) as depenency. This will provide accurate rendered data.
  - [ ] Provide preprocessing scripts for data and transform them into the required format for the network.
  ![](assets/carla-sim.gif)
  
2. Implement some parts of the neural network:
  - [ ] Implement steering in order to keep the center of the lane
  - [ ] Implement path following
  - [ ] Implement speed control
  - [ ] Implement road mask layer
  - [ ] Implement agent box output layer
  - [ ] Implement waypoint layer
  - [ ] Add other agents to input
  - [ ] Implement perception box output
  
3. Iterate from step 1 while adding more complexity
