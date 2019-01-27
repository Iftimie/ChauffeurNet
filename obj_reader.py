import numpy as np
from config import Config

def read_obj_file( path):
    # TODO might need to read the lines between vertices
    # TODO
    # TODO

    with open(path) as file:
        all_lines = file.readlines()
        file.close()

    all_objects = {}
    i = 0
    while i < len(all_lines):
        line = all_lines[i]

        if line[0] == "o":
            object_name = line.split(" ")[-1].replace("\n", "")
            object_vertices = []
            object_lines = []
            i += 1
            while i < len(all_lines) and all_lines[i][0] == "v":
                object_vertices.append(all_lines[i])
                i += 1
            while i < len(all_lines) and all_lines[i][0] == "l":
                object_lines.append(all_lines[i])
                i += 1
            all_objects[object_name] = {"verts": object_vertices,
                                        "lines": object_lines}
        i += 1

    for objname in all_objects.keys():
        object_vertices = all_objects[objname]["verts"]
        vertices_numeric = []
        for vertex in object_vertices:
            coords_str = vertex.replace("v ", "").replace("\n", "").split(" ") + ["1.0"]
            coords_numeric = [float(value) for value in coords_str]
            vertices_numeric.append(coords_numeric)
        vertices_numeric = np.array(vertices_numeric).T
        vertices_numeric[:3, :] *= Config.world_scale_factor
        all_objects[objname]["verts"] = vertices_numeric

    return all_objects


all_objects = read_obj_file("data\world.obj")
for objname in all_objects.keys():
    print(all_objects[objname])