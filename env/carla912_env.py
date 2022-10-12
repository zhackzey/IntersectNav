import glob
import os
import sys
import random
import numpy as np
import math as m
import csv

from PIL import Image
from collections import OrderedDict

import carla
from env.ego_car import EgoCar
from env.surroundings import Vehicles, Walkers
from env.configer import *
import queue
from env.utils import Waypoints, get_weather, get_area, INSIDE, get_task_type
from env.utils import FPS, draw_area, render_LIDAR, pre_process_lidar_912, get_raw_lidar
from env.PID.tools.misc import compute_magnitude_angle

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
        print('# Initializing Env 0.9.12')

        self.client = carla.Client("localhost", port)  # connect to server
        self.client.set_timeout(30.0)
        self.town = town
        self.client.load_world(town)
        self.world = self.client.get_world()
        self._settings = self.world.get_settings()
        self.ego_car = self.vehicles = self.walkers = None

        try:
            with time_limit(30):
                self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
                    no_rendering_mode=False,
                    synchronous_mode=True,
                    fixed_delta_seconds=1.0 / FPS))
        except TimeoutException:
            print("Error happened: apply carla settings.")
            self.success = False
        else:
            self.success = True

        self.traffic_manager = None
        try:
            self.traffic_manager = self.client.get_trafficmanager(8000)
            print("Use traffic manager")
        except:
            print("Not found API get_trafficmanager")
        else:
            pass
        if self.traffic_manager is not None:
            self.traffic_manager.set_global_distance_to_leading_vehicle(5)
            self.traffic_manager.global_percentage_speed_difference(50)
            self.traffic_manager.set_synchronous_mode(True)

        if CARLA_VERSION == '0.9.7':
            waypoint_file = 'waypoint_' + town[4:] + '.csv'
        elif CARLA_VERSION == '0.9.12':
            waypoint_file = 'waypoint_' + town[4:] + '_912.csv'
        else:
            raise NotImplementedError
        print("waypoint_file: ", waypoint_file)
        self.waypoints = Waypoints(os.path.join(os.path.dirname(__file__), waypoint_file))
        if debug:
            print('# waypoints number = ', self.waypoints.cnt)
            self.waypoints.render(self.world.debug)

        print("Done.")

    def reset(self, scene, debug=False, draw_area=False, manual_device=False, bird_view=False):
        if self.town != scene['town']:
            self.client.load_world(scene['town'])
            self.town = scene['town']
            self.world = self.client.get_world()
            self._settings = self.world.get_settings()
        self.client.reload_world()
        self.world = self.client.get_world()
        self.world.apply_settings(carla.WorldSettings(  # set synchronous mode
            no_rendering_mode=False,
            synchronous_mode=True,
            fixed_delta_seconds=1.0 / FPS))
        try:
            self.traffic_manager = self.client.get_trafficmanager(8000)
            print("Use traffic manager")
        except:
            print("Not found API get_trafficmanager")
        else:
            pass
        if self.traffic_manager is not None:
            self.traffic_manager.set_global_distance_to_leading_vehicle(5)
            self.traffic_manager.global_percentage_speed_difference(50)
            self.traffic_manager.set_synchronous_mode(True)

        self.scene = scene
        self.area = get_area(self.scene)

        self.ego_car = EgoCar(self.world, self.client, self.scene, self.waypoints, manual_device)
        self.vehicles = Vehicles(self.world, self.client, self.scene, self.ego_car)
        self.walkers = Walkers(self.world, self.client, self.scene, self.waypoints, self.ego_car)
        # print(f"weather: {self.scene['weather']}")
        self.world.set_weather(get_weather(self.scene['weather']))
        if CARLA_VERSION == '0.9.12':
            self.set_traffic_light_before()
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
        if bird_view:
            configer = Configer(scene['scene_id'])
            # generate a bird view camera sensor at the intersection center
            bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            bp.set_attribute('image_size_x', '720')
            bp.set_attribute('image_size_y', '720')
            bp.set_attribute('fov', '90')

            with open('env/waypoint_' + scene['town'][-2:] + '.csv') as f_raw:
                f = csv.reader(f_raw)
                for item in f:
                    if item[0] == str(configer.ped_center):
                        location = carla.Location(x=float(item[1]), y=float(item[2]), z=float(item[3]) + 20)  # 17
            self.bdv_sensor = self.world.spawn_actor(bp, carla.Transform(location, carla.Rotation(-90, 0, -90)))
            self.bdv_photos = queue.Queue()
            self.bdv_sensor.listen(self.bdv_photos.put)
        self.frame = self.start_frame = self.world.tick()

        self.reset_metrics()
        data = self.ego_car.get_sensors(self.frame)

        if bird_view:
            # retrieve bird view image from bdv_sensor
            event = self.bdv_photos.get()
            assert event.frame == self.frame
            event.convert(carla.ColorConverter.Raw)
            bdv_image = np.array(event.raw_data)
            bdv_image = bdv_image.reshape((720, 720, 4))[:, :, :3]

        state = self.get_state(data)
        info = self.get_info(data, state)
        if not bird_view:
            return state, info
        else:
            return state, info, bdv_image

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

    def step(self, action, lateral, longitude, debug=False, bird_view=False, steer_noise_inject=None):
        assert action.shape == (2,) or action.shape == (3, )
        if action.shape == (2, ):
            steer = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], 0.0, 1.0) if action[1] > 0.0 else 0.0
            brake = np.clip(abs(action[1]), 0.0, 1.0) if action[1] < 0.0 else 0.0
        elif action.shape == (3, ):
            steer = np.clip(action[0], -1.0, 1.0)
            throttle = np.clip(action[1], 0.0, 1.0)
            brake = np.clip(action[2], 0.0, 1.0)
        else:
            raise NotImplementedError
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=False)

        self.ego_car.step(control, lateral, longitude, steer_noise_inject=steer_noise_inject)

        self.vehicles.step()
        self.walkers.step()
        self.frame = self.world.tick()

        data = self.ego_car.get_sensors(self.frame)
        if bird_view:
            # retrieve bird view image from bdv_sensor
            event = self.bdv_photos.get()
            assert event.frame == self.frame
            event.convert(carla.ColorConverter.Raw)
            bdv_image = np.array(event.raw_data)
            bdv_image = bdv_image.reshape((720, 720, 4))[:, :, :3]
        state = self.get_state(data)
        info = self.get_info(data, state)
        if not debug:
            reward, done, error = self.get_reward_done(data, info)  # update self.res
        else:
            reward, done, error, disrupt_transforms = self.get_reward_done(data, info, True)
        if not debug:
            return state, reward, done, info, error
        else:
            if not bird_view:
                return state, reward, done, info, error, disrupt_transforms
            else:
                return state, reward, done, info, error, disrupt_transforms, bdv_image

    def idle_step(self):
        control = carla.VehicleControl(steer=0, throttle=0.2, brake=0, reverse=False)
        self.ego_car.vehicle.apply_control(control)
        self.vehicles.step()
        self.walkers.step()
        self.frame = self.world.tick()
        data = self.ego_car.get_sensors(self.frame)
        state = self.get_state(data)
        info = self.get_info(data, state)
        return state, info

    def get_state(self, data, debug=False):
        """return (image, lidar, measure, command)"""
        rgb = data['FrontRGB']
        rgb = rgb[115: 510, :]
        rgb = np.array(Image.fromarray(rgb).resize((200, 88)))

        dep = data['FrontDepth']
        dep = dep[115: 510, :]
        dep = np.array(Image.fromarray(dep).resize((200, 88)))

        sem = data['FrontSemantic']
        sem = sem[115: 510, :]
        sem = np.array(Image.fromarray(sem).resize((200, 88)))  # haven't / 255, in order to save menmory

        points = data['Lidar'][0]
        lidar = pre_process_lidar_912(points)
        if debug:
            render_LIDAR(points, data['Lidar'][1], self.world.debug)

        measure = []
        measure.append(data['speed'] / 30.0)
        measure.append(data['min_dis'])
        measure.append(data['angle_diff'] / 1.57)
        measure.append(data['dis_diff'])
        measure.extend(self.scene['lane_type'])  # (3,)
        measure = np.array(measure)

        command = data['command']
        location = [data['location'].x, data['location'].y, data['location'].z]
        rotation = [data['rotation'].pitch, data['rotation'].roll, data['rotation'].yaw]

        self.res['total_min_dis'] += abs(data['min_dis'])

        return [rgb, dep, sem, lidar, measure, command, location, rotation]

    def get_info(self, data, state):
        info = {}
        info['FrontRGB'] = data['FrontRGB']
        info['FrontDepth'] = data['FrontDepth']
        info['FrontSemantic'] = data['FrontSemantic']
        info['small_semantic'] = state[0]  # 88 * 200
        info['Lidar'] = get_raw_lidar(data['Lidar'][0])

        info['a_t'] = [data['control'].steer, data['control'].throttle, data['control'].brake]
        info['location'] = data['location']
        info['rotation'] = data['rotation']
        return info

    def get_reward_done(self, data, info, debug=False):
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
            tolerance = 5 if self.scene['branch'] == 0 else 10
            if self.res['invasion_time'] >= tolerance:  # lane invasion for too many times
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
        if abs(info['a_t'][0]) > 0.4:
            self.res['total_ego_jerk'] += 1

        if abs(info['a_t'][1]) > 0.9:
            self.res['total_ego_jerk'] += 1
        if not debug:
            self.res['total_other_jerk'] += self.walkers.get_disruption()
        else:
            disrupt_cnt, disrupt_transforms = self.walkers.get_disruption(True)
            self.res['total_other_jerk'] += disrupt_cnt
        if not debug:
            return np.array(reward), done, error  # rwd: (3,), done: bool
        else:
            return np.array(reward), done, error, disrupt_transforms

    def get_rule_based_control(self):
        return self.ego_car.get_pid_control()

    def destroy(self):
        # for x in [self.vehicles, self.walkers, self.ego_car]:
        #     if x:
        #         x.destroy()
        if CARLA_VERSION == '0.9.12':
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.no_rendering_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
            if self.traffic_manager is not None:
                self.traffic_manager.set_synchronous_mode(False)

        if self.vehicles:
            print('# destroy vehicles')
            self.vehicles.destroy()
        if self.walkers:
            print('# destroy walkers')
            self.walkers.destroy()
        if self.ego_car:
            print('# destroy ego car')
            self.ego_car.destroy()
        if hasattr(self, 'bdv_sensor'):
            if self.bdv_sensor:
                self.bdv_sensor.destroy()
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

    def set_traffic_light_before(self):
        lights_list = self.world.get_actors().filter("*traffic_light*")
        min_angle = 180.0
        sel_magnitude = 0.0
        sel_traffic_light = None
        for traffic_light in lights_list:
            loc = traffic_light.get_location()
            magnitude, angle = compute_magnitude_angle(loc, self.ego_car.vehicle.get_location(),
                                                       self.ego_car.vehicle.get_transform().rotation.yaw)
            if magnitude < 60.0 and angle < min(25.0, min_angle):
                sel_magnitude = magnitude
                sel_traffic_light = traffic_light
                min_angle = angle

        if sel_traffic_light is not None:
            print('traffic light === Magnitude = {} | Angle = {} | ID = {}'.format(
                sel_magnitude, min_angle, sel_traffic_light.id))
            sel_traffic_light.set_state(carla.TrafficLightState.Green)
            sel_traffic_light.freeze(True)