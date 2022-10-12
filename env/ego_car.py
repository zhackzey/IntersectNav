import glob
import os
import sys
import math
import random
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import copy

from env.sensors import camera_list, CameraSensor, CollisionSensor, LaneInvasionSensor
from env.utils import draw_waypoint_union, get_closet_wp, get_polygan

from env.PID.agent.basic_agent import BasicAgent
from env.PID.planner.local_planner import RoadOption
from env.SteeringWheel import steeringwheel
from env.configer import CARLA_VERSION
from env.PID.agent.utils import get_angle


class EgoCar:

    def __init__(self, world, client, scene, waypoints, manual_device=False):
        self.manual_device = manual_device
        self.world = world
        self.client = client
        self.scene = scene
        self.waypoints = waypoints

        self.start = self.waypoints.get_transform(self.scene['start'])
        self.end = self.waypoints.get_transform(self.scene['end'])

        bp = self.world.get_blueprint_library().filter("vehicle.audi.a2")[0]
        self.vehicle = self.world.spawn_actor(bp, self.start)

        self.agent = BasicAgent(self.vehicle)
        if self.manual_device:
            self.steeringWheel = steeringwheel()
        self.world.tick()

        self.path = self.agent.set_destination(self.end.location, draw_navi=False)

        self.sensors = []  # camera, Lidar, collision, lane invasion
        self.control = None
        self.param = {
            'acc_decrease': 0.0,
            'acc_maintain': 0.4,
            'acc_accelerate': 1.0,
            'steer_left': -0.1,
            'steer_right': 0.1
        }

    def get_setup(self):

        start_point = np.array([self.start.location.x, self.start.location.y, self.start.location.z])
        end_point = np.array([self.end.location.x, self.end.location.y, self.end.location.z])
        path = np.array([np.array([self.path[i][0].transform.location.x, self.path[i][0].transform.location.y,
                                   self.path[i][0].transform.location.z]) for i in range(len(self.path))])

        return start_point, end_point, path

    def set_sensors(self):
        for camera in camera_list:
            self.sensors.append(CameraSensor(self.world, self.vehicle, camera))

        self.sensors.append(CollisionSensor(self.world, self.vehicle, 'Collision'))
        self.sensors.append(LaneInvasionSensor(self.world, self.vehicle, 'LaneInvasion'))

    def step(self, control, lateral, longitude):
        # change traffic light green
        if CARLA_VERSION == '0.9.7':
            lights_list = self.world.get_actors().filter("*traffic_light*")
            for light in lights_list:
                light.set_state(carla.TrafficLightState.Green)
        elif self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
        else:
            pass
        # control to be applied of ego agent
        self.control = control
        # get control of PID and human
        manual_control = PID_control = self.agent.run_step(debug=False)
        if self.manual_device:
            manual_control = self.steeringWheel.get_control()
        # change lateral control according to lateral tag
        if lateral.startswith('PID'):
            # PID
            self.control.steer = PID_control.steer
        elif lateral.startswith('manual'):
            # manual
            self.control.steer = manual_control.steer
        else:
            # NN
            pass
        # change longitude control according to longitude tag
        if longitude.startswith('PID'):
            # PID
            self.control.throttle, self.control.brake = PID_control.throttle, PID_control.brake
        elif longitude.startswith('manual'):
            # manual
            self.control.throttle, self.control.brake = manual_control.throttle, manual_control.brake
        else:
            # NN
            pass

        self.vehicle.apply_control(self.control)

    def get_pid_control(self):
        if CARLA_VERSION == '0.9.7':
            lights_list = self.world.get_actors().filter("*traffic_light*")
            for light in lights_list:
                light.set_state(carla.TrafficLightState.Green)
        elif self.vehicle.is_at_traffic_light():
            traffic_light = self.vehicle.get_traffic_light()
            if traffic_light.get_state() == carla.TrafficLightState.Red:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.freeze(True)
        else:
            pass
        PID_control = self.agent.run_step(debug=False)
        return [PID_control.steer, PID_control.throttle, PID_control.brake]

    def get_sensors(self, frame0, debug=False):
        data = {sensor.name: sensor.get_data(frame0) for sensor in self.sensors}

        v = self.vehicle.get_velocity()
        data['speed'] = 3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)  # km/h

        data['velocity'] = v
        current = self.vehicle.get_transform()
        data['location'] = current.location
        data['rotation'] = current.rotation

        dx = self.end.location.x - current.location.x
        dy = self.end.location.y - current.location.y
        fwd = current.get_forward_vector()
        # data['angle_diff'] = math.radians(math.degrees(math.atan2(dy, dx)) - current.rotation.yaw)
        data['angle_diff'] = get_angle([dy, dx], [fwd.y, fwd.x])

        tot_x = self.end.location.x - self.start.location.x
        tot_y = self.end.location.y - self.start.location.y
        data['dis_diff'] = math.sqrt(dx ** 2 + dy ** 2) / math.sqrt(tot_x ** 2 + tot_y ** 2)

        if self.control:
            data['control'] = self.control
        else:
            data['control'] = self.vehicle.get_control()

        # find the nearest waypoint to current location in the planned path
        neighbor_wp, next_wp, min_dis = get_closet_wp(self.path, current.location, self.world.debug, draw_dis=False)
        data['min_dis'] = min_dis

        lat_cmd_dict = {RoadOption.LANEFOLLOW: 0, RoadOption.LEFT: 1, RoadOption.RIGHT: 2, RoadOption.STRAIGHT: 3}
        for i in range(len(self.path)):
            if self.path[i][0] == neighbor_wp:
                rule_control = self.get_auto_control()
                if self.path[i][1] in [RoadOption.LEFT, RoadOption.RIGHT, RoadOption.STRAIGHT]:
                    lat_cmd = lat_cmd_dict[self.path[i][1]]
                else:
                    if rule_control[0] > self.param['steer_right']:
                        lat_cmd = 2
                    elif rule_control[0] < self.param['steer_left']:
                        lat_cmd = 1
                    else:
                        lat_cmd = lat_cmd_dict[self.path[i][1]]
                if rule_control[1] - rule_control[2] < self.param['acc_decrease']:
                    lon_cmd = 0
                elif rule_control[1] - rule_control[2] < self.param['acc_maintain']:
                    lon_cmd = 1
                else:
                    lon_cmd = 2
                data['command'] = [lat_cmd, lon_cmd]
                break

        if debug:
            print('current yaw:', current.rotation.yaw)
            print('taget yaw:', math.degrees(math.atan2(dy, dx)))
            print('angle_diff', data['angle_diff'])

            print("minmum distance: ", min_dis)

            print('speed', data['speed'])
            print('dis_diff', data['dis_diff'])
            print('command', data['command'])

        return data

    def destroy(self):
        for x in self.sensors:
            x.destroy()
        self.client.apply_batch([carla.command.DestroyActor(x) for x in [self.vehicle]])
