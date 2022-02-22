import glob
import os
import sys
from collections import OrderedDict

import numpy as np
from PIL import Image

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from env.ego_car import EgoCar
from env.surroundings import Vehicles, Walkers

from env.utils import Waypoints, get_weather, get_area, INSIDE, get_task_type
from env.utils import FPS, render_LIDAR, pre_process_lidar

import signal
from contextlib import contextmanager


class TimeoutException(Exception): pass


@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


class Env:

    def __init__(self, port, debug=False, town='Town03'):
        print('# Initializing Env')

        self.client = carla.Client("localhost", port)  # connect to server
        self.client.set_timeout(4.0)
        self.client.load_world(town)
        self.world = self.client.get_world()
        self._settings = self.world.get_settings()

        try:
            with time_limit(10):
                self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
                    no_rendering_mode=False,
                    synchronous_mode=True,
                    fixed_delta_seconds=1. / FPS))
        except TimeoutException:
            print("Error happened: apply carla settings.")
            self.success = False
        else:
            self.success = True
        self.waypoints = Waypoints(os.path.join(os.path.dirname(__file__), 'waypoint_' + town[4:] + '.csv'))
        if debug:
            print('# waypoints number = ', self.waypoints.cnt)
            self.waypoints.render(self.world.debug)

        self.ego_car = self.vehicles = self.walkers = None
        print("Done.")

    def reset(self, scene, debug=False, draw_area=False, manual_device=False):
        self.scene = scene
        self.area = get_area(self.scene)
        self.world.set_weather(get_weather(self.scene['weather']))

        self.ego_car = EgoCar(self.world, self.client, self.scene, self.waypoints, manual_device)
        self.vehicles = Vehicles(self.world, self.client, self.scene, self.ego_car)
        self.walkers = Walkers(self.world, self.client, self.scene, self.waypoints, self.ego_car)

        if debug:
            spector = self.waypoints.get_transform(scene['ped_center'])
            spector.location.z = 30
            spector.rotation.pitch = -90
            self.world.get_spectator().set_transform(spector)

        self.vehicles.start()
        self.walkers.start()

        for _ in range(self.scene['Wait_ticks']):
            self.world.tick()

        self.ego_car.set_sensors()  # sensor: camera, Lidar, collison, lane invasion, info
        self.frame = self.start_frame = self.world.tick()

        self.reset_metrics()
        data = self.ego_car.get_sensors(self.frame)
        state = self.get_state(data)
        info = self.get_info(data, state)

        return state, info

    def reset_metrics(self):
        self.dict = OrderedDict()
        self.res = self.dict

        self.res['success'] = False
        self.res['time_out'] = False
        self.res['lane_invasion'] = False
        self.res['collision'] = False
        self.res['TooFar'] = False
        self.res['TooMuchAngle'] = False
        self.res['invasion_time'] = 0

        self.res['total_ego_jerk'] = 0
        # self.res['mean_ego_jerk'] = 0.0

        self.res['total_other_jerk'] = 0
        # self.res['mean_other_jerk'] = 0.0

        self.res['total_min_dis'] = 0.0
        # self.res['mean_min_dis'] = 0.0

        self.res['dis_to_destination'] = 0
        self.res['vertical_dist'] = 0
        self.res['delta_angle'] = 0
        self.res['wrong_direction'] = False

    def step(self, action, lateral, longitude):
        # TODO: judge the dimension of action; if dim = 3, the following code should be modified
        assert action.shape == (2,) or action.shape == (3,)
        if action.shape == (2,):
            steer = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], 0.0, 1.0) if action[1] > 0.0 else 0.0
            brake = np.clip(abs(action[1]), 0.0, 1.0) if action[1] < 0.0 else 0.0
        elif action.shape == (3,):
            steer = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], 0.0, 1.0)
            brake = np.clip(action[2], 0.0, 1.0)
        else:
            raise NotImplementedError
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=False)

        self.ego_car.step(control, lateral, longitude)

        self.vehicles.step()
        self.walkers.step()
        self.frame = self.world.tick()

        data = self.ego_car.get_sensors(self.frame)
        state = self.get_state(data)
        info = self.get_info(data, state)
        reward, done, error = self.get_reward_done(data, info)  # update self.res

        return state, reward, done, info, error

    def get_state(self, data, debug=False):
        """return (image, lidar, measure, command)"""
        rgb = data['FrontRGB']
        rgb = rgb[115: 510, :]
        rgb = np.array(Image.fromarray(rgb).resize((200, 88)))

        points = data['Lidar'][0]
        lidar, lidar_raw = pre_process_lidar(points)
        if debug:
            render_LIDAR(points, data['Lidar'][1], self.world.debug)

        measure = []
        measure.append(data['speed'] / 30.0)
        measure.append(data['min_dis'])
        measure.append(data['angle_diff'] / 1.57)
        measure.append(data['dis_diff'])
        measure = np.array(measure)

        command = data['command']
        location = [data['location'].x, data['location'].y, data['location'].z]
        rotation = [data['rotation'].pitch, data['rotation'].roll, data['rotation'].yaw]

        self.res['total_min_dis'] += abs(data['min_dis'])

        return (rgb, lidar, measure, command, location, rotation)

    def get_info(self, data, state):
        info = {}
        info['FrontRGB'] = data['FrontRGB']

        info['a_t'] = [data['control'].steer, data['control'].throttle, data['control'].brake]
        info['location'] = data['location']
        info['rotation'] = data['rotation']
        return info

    def get_reward_done(self, data, info):
        reward = []
        done = False
        error = 0

        reward.append(data['Collision'][0] > 0.0)
        reward.append(len(data['LaneInvasion']) > 0)

        if data['Collision'][0] > 0.0:
            done = True
            error = 1  # collision
            self.res['collision'] = True

        if len(data['LaneInvasion']) > 0:
            self.res['invasion_time'] += 1
            if self.res['invasion_time'] >= 5:  # lane invasion for too many times
                done = True
                # error = 2 # lane invasion
                self.res['lane_invasion'] = True
            error = 2

        reward.append(0)  # don't success
        if not INSIDE(data['location'], self.area):
            done = True
            task_type = get_task_type(data['location'], self.area, self.scene['task_type'])

            if self.scene['branch'] != task_type:
                error = 3  # go through the crossing in wrong direction
                self.res['wrong_direction'] = True
            else:
                self.res['success'] = True
                reward[-1] = 1  # success

        # Comfort
        if abs(info['a_t'][0]) > 0.9:
            self.res['total_ego_jerk'] += 1

        if abs(info['a_t'][1]) > 0.9:
            self.res['total_ego_jerk'] += 1

        self.res['total_other_jerk'] += self.walkers.get_disruption()

        return np.array(reward), done, error  # rwd: (3,), done: bool

    def get_rule_based_control(self):
        return self.ego_car.get_pid_control()

    def destroy(self):
        for x in [self.vehicles, self.walkers, self.ego_car]:
            if x:
                x.destroy()
        self.ego_car = self.vehicles = self.walkers = None

    def close(self, expected_end_steps):
        self.destroy()

        self.res['tot_step'] = self.frame - self.start_frame
        if self.res['tot_step'] == expected_end_steps:
            self.res['time_out'] = True
        return self.res

    def draw_point(self, end):
        debug = self.world.debug
        location = ['loc_x', 'loc_y', 'loc_z']
        x, y, z = self.waypoints.locs[end]

        loc = carla.Location(x, y, z)
        debug.draw_point(loc, size=1, life_time=8, persistent_lines=True)

    def __del__(self):
        self.destroy()