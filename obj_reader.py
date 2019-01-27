import numpy as np



all_objects = read_obj_file("world.obj")
for objname in all_objects.keys():
    print(all_objects[objname])