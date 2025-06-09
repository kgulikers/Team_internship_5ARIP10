## Credits and dependencies

This project uses the simulation setup and robot model from the official Avular Origin One simulation repository:
[https://github.com/avular-robotics/avular_origin_simulation](https://github.com/avular-robotics/avular_origin_simulation)

The following files were adapted to create a custom simulation scenario, to resemble the layout used during training in Isaac Lab:

### Custom Room Model
```bash
RL_container/avular_origin_simulation/origin_one_gazebo/models/model.sdf
```
A new SDF model was created from scratch to define a  10×10 meter room, with two static box-shaped obstacles with different dimensions.

### Modified Launch File
```bash
RL_container/avular_origin_simulation/origin_one_gazebo/launch/ty_test_area.launch.py
```
The launch file was modified to ensure that the Origin One robot spawns directly inside the custom room model, so the robot begins in a valid position relative to the surrounding walls and goal during each simulation run, in a similar spot as the Isaac Lab simulation. 

### Adjusted World Configuration
```bash
RL_container/avular_origin_simulation/origin_one_gazebo/worlds/TY_test_area.world
```
Minor edits were made to include a goal marker model into the scene and adjust the room's position on the proper Z plane. 
