import glob
import os
import sys
import math
import numpy as np
import pandas as pd
import cv2
import h5py

from collections import OrderedDict
from env.configer import PATH_TO_ROUTES, PATH_TO_MAPS, width_in_pixels, pixels_ev_to_bottom, pixels_per_meter, CARLA_VERSION

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla



###################################
#   Parameters
###################################
FPS = 20 if CARLA_VERSION == '0.9.7' else 30
CrossingRate = 0.0  # allow walkers to cross road
SafyDis = 5.0       # The safy distance between spawned surrounding car and ego car



###################################
#   Waypoints
###################################

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

####################################
#    Helper Functions     
####################################


def get_weather(weather):
    '''
    Carla 0.9.7: ClearNoon, CloudyNoon, WetNoon, WetCloudyNoon, SoftRainNoon, MidRainyNoon, HardRainNoon, ClearSunset,
    CloudySunset, WetSunset, WetCloudySunset, SoftRainSunset, MidRainSunset, HardRainSunset

    Carla 0.9.12: ClearNight, ClearNoon, ClearSunset, CloudyNight, CloudyNoon, CloudySunset, HardRainNight,
    HardRainNoon, HardRainSunset, MidRainSunset, MidRainyNight, MidRainyNoon, SoftRainNight, SoftRainNoon, SoftRainSunset,
    WetCloudyNight, WetCloudyNoon, WetCloudySunset, WetNight, WetNoon, WetSunset
    '''
    weather_paramters = carla.WeatherParameters
    return getattr(weather_paramters, weather)


def get_PED_area(scene, waypoints):
    origin = waypoints.locs[scene['ped_center']]
    PED_AREA = {
        'MINX': origin[0] - scene['ped_range'],
        'MAXX': origin[0] + scene['ped_range'],
        'MINY': origin[1] - scene['ped_range'],
        'MAXY': origin[1] + scene['ped_range'],
    }
    return PED_AREA


def get_area(scene):
    AREA = {
        'MINX': scene['area'][0],
        'MAXX': scene['area'][1],
        'MINY': scene['area'][2],
        'MAXY': scene['area'][3],
    }
    return AREA


def INSIDE(loc, AREA):
    return (loc is not None) and loc.x >= AREA['MINX'] and loc.x <= AREA['MAXX'] and \
        loc.y >= AREA['MINY'] and loc.y <= AREA['MAXY'] 


def get_distance_to_area(loc, AREA):
    area_center_x = (AREA['MINX'] + AREA['MAXX']) / 2.0
    area_center_y = (AREA['MINY'] + AREA['MAXY']) / 2.0
    return math.sqrt((loc.x - area_center_x) ** 2 + (loc.y - area_center_y) ** 2)

def get4areas(AREA):
    areas=[]
    area1={
        'MINX': AREA['MINX']+15,
        'MAXX': AREA['MINX']+20,
        'MINY': AREA['MINY'],
        'MAXY': AREA['MINY']+20,
    }
    area2={
        'MINX': AREA['MINX']+15,
        'MAXX': AREA['MINX']+20,
        'MINY': AREA['MAXY']-20,
        'MAXY': AREA['MAXY'],
    }
    area3={
        'MINX': AREA['MAXX']-20,
        'MAXX': AREA['MAXX']-16,
        'MINY': AREA['MINY'],
        'MAXY': AREA['MINY']+20,
    }
    area4={
        'MINX': AREA['MAXX']-20,
        'MAXX': AREA['MAXX']-16,
        'MINY': AREA['MAXY']-20,
        'MAXY': AREA['MAXY'],
    }
    areas.append(area1)
    areas.append(area2)
    areas.append(area3)
    areas.append(area4)
    return areas


def get_task_type(loc, area, task_type):
    if loc.x < area['MINX']:
        return task_type[0]

    if loc.x > area['MAXX']:
        return task_type[1]

    if loc.y < area['MINY']:
        return task_type[2]

    if loc.y > area['MAXY']:
        return task_type[3]
    
    assert False, 'loc must be in out of the area when calling this function'



######################################
#    Calc distance
######################################

def get_closet_wp(path, cur_loc, debug, draw_dis):

    lent = len(path)
    min_dis = 1e6
    min_ind = 0
    for i in range(lent):
        temp_loc = path[i][0].transform.location
        dis = (temp_loc.x - cur_loc.x) * (temp_loc.x - cur_loc.x) + (temp_loc.y - cur_loc.y) * (temp_loc.y - cur_loc.y)
        if dis < min_dis:
            min_dis = dis
            min_ind = i

    if draw_dis:
        temp_loc = path[min_ind][0].transform.location
        debug.draw_line(cur_loc, temp_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=1.0, persistent_lines=False)

    min_dis = math.sqrt(min_dis)

    x1 = path[min_ind+1][0].transform.location.x - path[min_ind][0].transform.location.x + 0.001
    y1 = path[min_ind+1][0].transform.location.y - path[min_ind][0].transform.location.y + 0.001

    x2 = cur_loc.x - path[min_ind][0].transform.location.x
    y2 = cur_loc.y - path[min_ind][0].transform.location.y 

    min_height = (x1 * y2 - y1 * x2) / math.sqrt(x1**2 + y1**2)
    #print('height', min_height)

    return path[min_ind][0], path[min_ind+1][0], min_height


def get_polygan(debug, veh_loc, yaw_loc, neigh_loc, next_loc, draw_dis):

    if draw_dis:
        debug.draw_line(veh_loc, yaw_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(yaw_loc, next_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(next_loc, neigh_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)
        debug.draw_line(neigh_loc, veh_loc, thickness=0.5, color=carla.Color(255, 0, 0), life_time=0.2, persistent_lines=False)

    vec_1 = (veh_loc.x - neigh_loc.x, veh_loc.y - neigh_loc.y)
    vec_2 = (next_loc.x - neigh_loc.x, next_loc.y - neigh_loc.y)
    vec_3 = (veh_loc.x - next_loc.x, veh_loc.y - next_loc.y)
    vec_4 = (veh_loc.x - yaw_loc.x, veh_loc.y - yaw_loc.y)
    vec_5 = (next_loc.x - yaw_loc.x, next_loc.y - yaw_loc.y)

    len_1 = math.sqrt(vec_1[0]**2 + vec_1[1]**2)
    len_2 = math.sqrt(vec_2[0]**2 + vec_2[1]**2)
    len_3 = math.sqrt(vec_3[0]**2 + vec_3[1]**2)
    len_4 = math.sqrt(vec_4[0]**2 + vec_4[1]**2)
    len_5 = math.sqrt(vec_5[0]**2 + vec_5[1]**2)

    circum_1 = len_1 + len_2 + len_3
    circum_2 = len_3 + len_4 + len_5

    if len_1 < 0.1:
        area_1 = 0.
    else:
        area_1 = math.sqrt(circum_1/2 * (circum_1/2 - len_1) * (circum_1/2 - len_2) * (circum_1/2 - len_3))
    if len_5 < 0.1:
        area_2 = 0.
    else:
        area_2 = math.sqrt(circum_2/2 * (circum_2/2 - len_3) * (circum_2/2 - len_4) * (circum_2/2 - len_5))

    return area_1 + area_2



######################################
#    Render
######################################

def draw_waypoint_union(debug, l0, l1, to_show, color=carla.Color(0, 0, 255), lt=100):

    debug.draw_line( l0.transform.location + carla.Location(z=0.25), l1.transform.location + carla.Location(z=0.25),
        thickness=0.1, color=color, life_time=lt, persistent_lines=False)
    debug.draw_point(l1.transform.location + carla.Location(z=0.25), 0.01, color, lt, False)
    debug.draw_string(l1.transform.location, str(to_show), False, carla.Color(255, 162, 0), 200, persistent_lines=False)


def draw_area(debug, AREA, color = (0, 255, 0)):
    a0 = carla.Location(x = AREA['MINX'], y = AREA['MINY'], z = 5)
    a1 = carla.Location(x = AREA['MINX'], y = AREA['MAXY'], z = 5)
    a2 = carla.Location(x = AREA['MAXX'], y = AREA['MINY'], z = 5)
    a3 = carla.Location(x = AREA['MAXX'], y = AREA['MAXY'], z = 5)
    
    color = carla.Color(color[0], color[1], color[2])
    thickness = 1
    debug.draw_line(a0, a1, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a1, a3, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a3, a2, thickness = thickness, color = color, life_time = 100.)
    debug.draw_line(a2, a0, thickness = thickness, color = color, life_time = 100.)



######################################
#    Lidar
######################################

def pre_process_lidar_912(points):
    lidar_feature = [1.0] * 720
    i = 0
    for point in points:
        i = (i+1)%1440
        point = np.array([point[0], point[1]])
        rel_dis = get_distance(point)
        rel_deg = get_angle(point)

        if rel_deg < 720 and rel_dis < lidar_feature[rel_deg]:
            lidar_feature[rel_deg] = rel_dis

    # print("lidar_list: ", raw_lidar)
    # print("number of points: ", i)
    return np.array(lidar_feature)

def pre_process_lidar(points):
    lidar_feature = [1.0] * 720
    # print("points: ", points)
    raw_lidar = [np.array([-1.0, -1.0, -1.0])] * 1440
    i = 0
    for point in points:
        raw_lidar[i] = np.array([point[0], point[1], point[2]])
        i = (i+1)%1440
        point = np.array([point[0], point[1]])
        rel_dis = get_distance(point)
        rel_deg = get_angle(point)

        if rel_deg < 720 and rel_dis < lidar_feature[rel_deg]:
            lidar_feature[rel_deg] = rel_dis

    # print("lidar_list: ", raw_lidar)
    # print("number of points: ", i)
    return np.array(lidar_feature), np.array(raw_lidar)

def get_raw_lidar(points):
    with_intensity = False
    if points.shape[-1] == 3:
        raw_lidar = [np.array([-1.0, -1.0, -1.0])] * 1440
    elif points.shape[-1] == 4:
        raw_lidar = [np.array([-1.0, -1.0, -1.0, 0.0])] * 1440
        with_intensity = True
    else:
        raise NotImplementedError
    i = 0
    for point in points:
        if not with_intensity:
            raw_lidar[i] = np.array([point[0], point[1], point[2]])
        else:
            raw_lidar[i] = np.array([point[0], point[1], point[2], point[3]])
        i = (i+1)%1440
    # print("lidar_list: ", raw_lidar)
    # print("number of points: ", i)
    return np.array(raw_lidar)

def get_distance(point):
    d_x = point[0]
    d_y = point[1]
    dis = math.sqrt(d_x*d_x+d_y*d_y)
    if dis < 0.1:
        # print("Too close! Impossible!!", d_x, " ", d_y)
        return 0.1/40.0
    rel_dis = dis/40.0
    # print("rel_dis", rel_dis)
    rel_dis = min(1.0, rel_dis)

    return rel_dis

def get_angle(point):

    if point[0] - 0.0 < 1e-3 and point[0] > 0.0:
        point[0] = 1e-3
    if 0.0 - point[0] < 1e-3 and point[0] < 0.0:
        point[0] = -1e-3

    angle = math.atan2(point[1], point[0])
    while (angle > math.pi):
        angle -= 2 * math.pi
    while (angle < -math.pi):
        angle += 2 * math.pi
    assert angle>=-np.pi and angle<=np.pi
    degree = math.degrees(angle)
    # print("degree", degree)

    rel_degree = int((4*degree + 720.0) % 1440)
    # print("rel_degree", rel_degree)

    return rel_degree


def get_matrix(rotation):
    """
    Creates matrix from carla transform.
    """
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    matrix = np.matrix(np.identity(2))
    # matrix[0, 0] = c_y
    # matrix[0, 1] = -s_y
    # matrix[1, 0] = s_y
    # matrix[1, 1] = c_y
    matrix[0, 0] = -s_y
    matrix[0, 1] = -c_y
    matrix[1, 0] = c_y
    matrix[1, 1] = -s_y
    return matrix


def sensor_to_world(cords, sensor_transform):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor_transform.rotation)
    world_cords = np.dot(sensor_world_matrix, np.transpose(cords))
    world_cords[0,0] += sensor_transform.location.x
    world_cords[0,1] += sensor_transform.location.y
    world_cords = world_cords.tolist()
    return world_cords[0]


def render_LIDAR(points, lidar_transform, debug):
    for point in points:
        point = np.array([point[0], point[1]])
        rel_angle = get_angle(point)
        point_loc = sensor_to_world(point, lidar_transform)
        point_loc = carla.Location(x=point_loc[0], y=point_loc[1], z=1.5)
        if rel_angle<720:
            debug.draw_point(point_loc, 0.1, carla.Color(255, 162, 0), 0.05, False)


def get_intention(vec_1, vec_2):
    l_1 = math.sqrt(vec_1[0]**2+vec_1[1]**2)
    l_2 = math.sqrt(vec_2[0]**2+vec_2[1]**2)
    if l_1<0.1 or l_2<0.1:
        return -1.
    return (vec_1[0]*vec_2[0]+vec_1[1]*vec_2[1])/l_1/l_2


def loc_global_to_ref(target_loc_in_global, ref_trans_in_global):
    """
    :param target_loc_in_global: carla.Location in global coordinate (world, actor)
    :param ref_trans_in_global: carla.Transform in global coordinate (world, actor)
    :return: carla.Location in ref coordinate
    """
    x = target_loc_in_global.x - ref_trans_in_global.location.x
    y = target_loc_in_global.y - ref_trans_in_global.location.y
    z = target_loc_in_global.z - ref_trans_in_global.location.z
    vec_in_global = carla.Vector3D(x=x, y=y, z=z)
    vec_in_ref = vec_global_to_ref(vec_in_global, ref_trans_in_global.rotation)

    target_loc_in_ref = carla.Location(x=vec_in_ref.x, y=vec_in_ref.y, z=vec_in_ref.z)
    return target_loc_in_ref


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x],
                                 [target_vec_in_global.y],
                                 [target_vec_in_global.z]])
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
    return target_vec_in_ref


def rot_global_to_ref(target_rot_in_global, ref_rot_in_global):
    target_roll_in_ref = cast_angle(target_rot_in_global.roll - ref_rot_in_global.roll)
    target_pitch_in_ref = cast_angle(target_rot_in_global.pitch - ref_rot_in_global.pitch)
    target_yaw_in_ref = cast_angle(target_rot_in_global.yaw - ref_rot_in_global.yaw)

    target_rot_in_ref = carla.Rotation(roll=target_roll_in_ref, pitch=target_pitch_in_ref, yaw=target_yaw_in_ref)
    return target_rot_in_ref

def rot_ref_to_global(target_rot_in_ref, ref_rot_in_global):
    target_roll_in_global = cast_angle(target_rot_in_ref.roll + ref_rot_in_global.roll)
    target_pitch_in_global = cast_angle(target_rot_in_ref.pitch + ref_rot_in_global.pitch)
    target_yaw_in_global = cast_angle(target_rot_in_ref.yaw + ref_rot_in_global.yaw)

    target_rot_in_global = carla.Rotation(roll=target_roll_in_global, pitch=target_pitch_in_global, yaw=target_yaw_in_global)
    return target_rot_in_global


def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix

def get_loc_rot_vel_in_ev(actor_list, ev_transform):
    location, rotation, absolute_velocity = [], [], []
    for actor in actor_list:
        # location
        location_in_world = actor.get_transform().location
        location_in_ev = loc_global_to_ref(location_in_world, ev_transform)
        location.append([location_in_ev.x, location_in_ev.y, location_in_ev.z])
        # rotation
        rotation_in_world = actor.get_transform().rotation
        rotation_in_ev = rot_global_to_ref(rotation_in_world, ev_transform.rotation)
        rotation.append([rotation_in_ev.roll, rotation_in_ev.pitch, rotation_in_ev.yaw])
        # velocity
        vel_in_world = actor.get_velocity()
        vel_in_ev = vec_global_to_ref(vel_in_world, ev_transform.rotation)
        absolute_velocity.append([vel_in_ev.x, vel_in_ev.y, vel_in_ev.z])
    return location, rotation, absolute_velocity

def cast_angle(x):
    # cast angle to [-180, +180)
    return (x+180.0)%360.0-180.0


def get_world_offset(town):
    if type(town) is int:
        assert town in [3, 5]
        maps_h5_path = PATH_TO_MAPS / f'Town0{town}.h5'
    elif type(town) is str:
        assert town in ['Town03', 'Town05']
        maps_h5_path = PATH_TO_MAPS / f'{town}.h5'
    else:
        raise TypeError
    with h5py.File(maps_h5_path, 'r', libver='latest', swmr=True) as hf:
        return np.array(hf.attrs['world_offset_in_meters'], dtype=np.float32)


def world_to_pixel(location, world_offset, projective=False):
    """Converts the world coordinates to pixel coordinates"""
    x = pixels_per_meter * (location[0] - world_offset[0])
    y = pixels_per_meter * (location[1] - world_offset[1])

    if projective:
        p = np.array([x, y, 1], dtype=np.float32)
    else:
        p = np.array([x, y], dtype=np.float32)
    return p


def get_warp_transform(ev_loc, ev_rot, world_offset):
    ev_loc_in_px = world_to_pixel(ev_loc, world_offset)
    yaw = np.deg2rad(ev_rot[2])

    forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
    right_vec = np.array([np.cos(yaw + 0.5 * np.pi), np.sin(yaw + 0.5 * np.pi)])

    bottom_left = ev_loc_in_px - pixels_ev_to_bottom * forward_vec - (0.5 * width_in_pixels) * right_vec
    top_left = ev_loc_in_px + (width_in_pixels - pixels_ev_to_bottom) * forward_vec - (0.5 * width_in_pixels) * right_vec
    top_right = ev_loc_in_px + (width_in_pixels - pixels_ev_to_bottom) * forward_vec + (0.5 * width_in_pixels) * right_vec

    src_pts = np.stack((bottom_left, top_left, top_right), axis=0).astype(np.float32)
    dst_pts = np.array([[0, width_in_pixels - 1],
                        [0, 0],
                        [width_in_pixels - 1, 0]], dtype=np.float32)
    return cv2.getAffineTransform(src_pts, dst_pts)


def get_bev_route_mask(ev_loc, ev_rot, world_offset, route):
    def _get_closest_wp(ev_loc, route):
        min_dis = 1e20
        min_idx = -1
        for i in range(len(route)):
            if (ev_loc[0] - route[i][0]) ** 2 + (ev_loc[1] - route[i][1]) ** 2 < min_dis:
                min_dis = (ev_loc[0] - route[i][0]) ** 2 + (ev_loc[1] - route[i][1]) ** 2
                min_idx = i
        return min_idx
    M_warp = get_warp_transform(ev_loc, ev_rot, world_offset)
    # x, y, z, cmd
    closest_wp_idx = _get_closest_wp(ev_loc, route)
    wps_to_draw = route[closest_wp_idx:, :2]
    route_mask = np.zeros([width_in_pixels, width_in_pixels], dtype=np.uint8)
    route_in_pixel = np.array([[world_to_pixel(location, world_offset, pixels_per_meter)]
                               for location in wps_to_draw])
    route_warped = cv2.transform(route_in_pixel, M_warp)
    cv2.polylines(route_mask, [np.round(route_warped).astype(np.int32)], False, 1, thickness=16)
    return route_mask
