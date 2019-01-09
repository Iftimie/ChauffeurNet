
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
    test_waypoint_idx = 2
    horizon_past = 24
    horizon_future = 8
    num_skip_poses = 14
    num_past_poses = horizon_past * num_skip_poses
    num_future_poses = horizon_future * num_skip_poses
    rnn_num_channels = 32

    #Camera params
    cam_height = -1200

    #Vehicle params
    vehicle_x = 100     #initial vehicle location x
    displace_z = -100   #camera dispacement

    #Google path length
    path_future_len = 250

    #Outputs (don't take into account feature extractor
    #nn_outputs=["steering","waypoints"]
    nn_outputs=["waypoints" ,"speed"]

    #Dropout
    num_frames = 70
    dropout_prob = 0.5
    amplitude_range = (0.5/10, 2/10)

    normalizing_speed = 10.0
    max_speed = normalizing_speed