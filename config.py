
class Config:

    # Render resolution. H, W
    # r_res = (576, 768)
    r_res = (144, 192)
    o_res = (36, 48)
    scale_factor = r_res[0] / o_res[0]

    if scale_factor ==2:
        features_num_channels = 64
    if scale_factor == 4:
        features_num_channels = 128
    elif scale_factor == 8:
        features_num_channels = 256

    # ratio is used to scale opencv shapes such as circles, which in general are defined in pixels, and do not depend on window size
    # or camera parameters such as focal length
    # initially all variables where designed at VGA scale
    r_ratio = 640.0 / r_res[1]
    o_ratio = 640.0 / o_res[1]


    #Temporal part
    test_waypoint_idx_steer = 2
    test_waypoint_idx_speed = 5
    horizon_past = 8
    horizon_future = 8
    num_skip_poses = 14
    num_past_poses = horizon_past * num_skip_poses
    num_future_poses = horizon_future * num_skip_poses
    rnn_num_channels = 32

    #Camera params
    cam_height = -1200

    #Vehicle params
    vehicle_x = -80     #initial vehicle location x
    vehicle_z = -300     #initial vehicle location z
    displace_z = -100   #camera dispacement

    #Google path length
    path_future_len = 350

    #Outputs (don't take into account feature extractor
    #nn_outputs=["steering","waypoints", "speed"]
    nn_outputs=["waypoints" ]
    # nn_outputs=["waypoints", "speed"]

    #Dropout
    num_frames = 70
    dropout_prob = 0.2
    amplitude_range = (0.5/10, 2/10)

    normalizing_speed = 10.0
    max_speed = 6

    #the selected waypoint correspons to a maximum distance from car of test_waypoint_idx_speed * num_skip_poses * max_speed   at max speed
    #as waypoints get closer together near the intersection, the selected waypoint might not be sufficiently close to the car in order for it to decrease speed
    #thus we need to interpolate the distance to waypoint from (0, max_waypoint_distance) to (0, max_speed)
    max_waypoint_distance = (test_waypoint_idx_speed * num_skip_poses * max_speed) / 2
    min_waypoint_distance = 0 # this is for test_waypoint_idx_speed 5

    linux_env = False


    #from fucking blender obj export. I need to rotate and scale some things
    world_scale_factor = 70
    scale_x = -1