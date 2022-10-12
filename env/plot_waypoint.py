import glob
import os
import sys
import csv
import time
import random

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import pandas as pd
import numpy as np

from scenes_912 import *

color_lists = [carla.Color(255, 0, 0),
               carla.Color(0, 255, 0),
               carla.Color(47, 210, 231),
               carla.Color(0, 255, 255),
               carla.Color(255, 255, 0),
               carla.Color(255, 162, 0),
               carla.Color(255, 255, 255)]

class Waypoints:
    def __init__(self, csv_path):
        waypoints = pd.read_csv(csv_path)
        waypoints = waypoints.dropna()
        location = ['loc_x', 'loc_y', 'loc_z']
        rotation = ['pitch', 'yaw', 'roll']

        self.locs = np.array(waypoints[location])
        self.rots = np.array(waypoints[rotation])
        self.cnt = len(waypoints['loc_x'])

    def get_transform(self, id):
        x, y, z = self.locs[id]
        pitch, yaw, roll = self.rots[id]

        return carla.Transform(carla.Location(x, y, z), carla.Rotation(pitch, yaw, roll))

    def render(self, debug):
        for i in range(self.cnt):
            x, y, z = self.locs[i]
            loc = carla.Location(x, y, z)

            debug.draw_point(loc, persistent_lines=True)
            debug.draw_string(loc, str(i), False, carla.Color(255, 162, 0), 200, persistent_lines=False)

def get_scene_layout(carla_map, precision = 1.0):
    """
    Function to extract the full scene layout to be used as a full scene description to be
    given to the user
    :return: a dictionary describing the scene.
    """

    def _lateral_shift(transform, shift):
        transform.rotation.yaw += 90
        return transform.location + shift * transform.get_forward_vector()

    topology = [x[0] for x in carla_map.get_topology()]
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    # A road contains a list of lanes, a each lane contains a list of waypoints
    map_dict = dict()

    for waypoint in topology:
        waypoints = [waypoint]
        nxt = waypoint.next(precision)
        if len(nxt) > 0:
            nxt = nxt[0]
            while nxt.road_id == waypoint.road_id:
                waypoints.append(nxt)
                nxt = nxt.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                else:
                    break

        lane = {
            "waypoints": waypoints
        }

        if map_dict.get(waypoint.road_id) is None:
            map_dict[waypoint.road_id] = {}
        map_dict[waypoint.road_id][waypoint.lane_id] = lane

    # Generate waypoints graph
    waypoints_graph = dict()
    for road_key in map_dict:
        for lane_key in map_dict[road_key]:
            # List of waypoints
            lane = map_dict[road_key][lane_key]

            for i in range(0, len(lane["waypoints"])):
                # Waypoint Position
                wl = lane["waypoints"][i].transform.location

                # Waypoint Orientation
                wo = lane["waypoints"][i].transform.rotation

                # Waypoint dict
                waypoint_dict = {
                    "road_id": road_key,
                    "lane_id": lane_key,
                    "id": lane["waypoints"][i].id,
                    "position": wl,
                    "orientation": wo
                }
                waypoints_graph[map_dict[road_key][lane_key]["waypoints"][i].id] = waypoint_dict

    return waypoints_graph

def create_csv(path):
    with open(path, 'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["ID", "loc_x", "loc_y", "loc_z", "pitch", "yaw", "roll"]
        csv_write.writerow(csv_head)

def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def draw_waypoint(debug, map, precision, csv_path):
    create_csv(csv_path)
    graph = get_scene_layout(map, precision)
    cnt = 0
    for v in graph.values():
        loc = v["position"]
        idn = v["id"]
        rot = v["orientation"]
        row = [str(cnt), str(loc.x), str(loc.y), str(loc.z), str(rot.pitch), str(rot.yaw), str(rot.roll)]
        write_csv(csv_path, row)
        debug.draw_point(loc, persistent_lines=True,life_time=0)
        debug.draw_string(loc, str(cnt), False, carla.Color(255, 162, 0), 2000000, persistent_lines=False)
        cnt += 1

    print("number of points: ", cnt)

def draw_scene(debug, scene, waypoints):
    x, y, z = waypoints.locs[scene.ped_center]
    debug.draw_point(carla.Location(x, y, z + 1), persistent_lines=True, size=0.15, color=carla.Color(0, 0, 0), life_time=0)
    debug.draw_string(carla.Location(x, y, z + 1),  "Ped center %d" % scene.ped_center, False, color=carla.Color(0, 0, 0), life_time=200000)

    area_min_x, area_max_x, area_min_y, area_max_y = scene.area
    debug.draw_point(carla.Location(area_min_x, area_min_y, z + 1), persistent_lines=True, size=0.15, color=carla.Color(0,0,0), life_time=0)
    debug.draw_string(carla.Location(area_min_x, area_min_y, z + 1), 'min_x_y', False, carla.Color(255, 162, 0), 20000, persistent_lines=True)
    debug.draw_point(carla.Location(area_max_x, area_max_y, z + 1), persistent_lines=True, size=0.15, color=carla.Color(0,0,0), life_time=0)
    debug.draw_string(carla.Location(area_max_x, area_max_y, z + 1), 'max_x_y', False, carla.Color(255, 162, 0), 20000, persistent_lines=True)

    idx = 0
    for pose in scene.poses:
        st_x, st_y, st_z = waypoints.locs[pose]
        debug.draw_point(carla.Location(st_x, st_y, st_z), persistent_lines=True, size=0.15, color=color_lists[idx], life_time=0)
        debug.draw_string(carla.Location(st_x, st_y, st_z), 'pose%d' % pose, False, color_lists[idx], 20000, persistent_lines=True)
        for end in scene.ends[idx]:
            if end == -1:
                continue
            ed_x, ed_y, ed_z = waypoints.locs[end]
            debug.draw_point(carla.Location(ed_x, ed_y, ed_z), persistent_lines=True, size=0.15, color=color_lists[idx],
                             life_time=0)
            debug.draw_string(carla.Location(ed_x, ed_y, ed_z), 'end%d' % end, False, color_lists[idx], 20000,
                              persistent_lines=True)
        idx+=1

if __name__ == '__main__':


    # client = carla.Client("localhost", 2000)
    client = carla.Client("localhost", 2000)

    client.set_timeout(4.0)
    world = client.get_world()
    map = world.get_map()
    debug = world.debug
    csv_path = './waypoint_03_carla912.csv'

    town3_scenes = [scene0(), scene1(), scene2(), scene3()]
    town5_scenes = [scene4(), scene5()]
    waypoints = Waypoints('./env/waypoint_03.csv')
    # waypoints = Waypoints('./env/waypoint_05.csv')
    # waypoints = Waypoints('./env/waypoint_03_carla912.csv')

    for scene in town3_scenes:
        draw_scene(debug, scene, waypoints)

    spawn_points = map.get_spawn_points()
    for point in spawn_points:
        debug.draw_point(point.location, persistent_lines=True, size=0.15, color=carla.Color(0, 255, 0), life_time=0)
    # draw_waypoint(debug, map, 5.0, csv_path)