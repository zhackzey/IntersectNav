import glob
import os
import sys
import math
import queue
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from env.utils import FPS
from env.configer import CARLA_VERSION

camera_list = [
    {
        'name': 'Lidar',
        'type': 'sensor.lidar.ray_cast',
        'range' : 40.0,
        'channels': 1,
        'upper_fov': 0.0,
        'lower_fov': 0.0,
        'rotation_frequency': float(FPS),
        'points_per_second': FPS*360*4,
        'pos' : carla.Transform(carla.Location(x=1.7, z=1.12)),
        'convertor': None
    },
    {
       'name': 'FrontRGB',
       'type': 'sensor.camera.rgb',
       'width' : 1500,
       'height' : 650,
       'fov': 140.0,
       'pos' : carla.Transform(carla.Location(x=1.0, z=1.6)),
       'convertor': cc.Raw,
        # 'camera_exposure_mode': 'manual',
        # 'shutter_speed': 3000
        'iso': 100.0,
        "exposure_mode": "histogram",
        "exposure_compensation": 0.0,
        "exposure_min_bright": 10.0,
        "exposure_max_bright": 12.0,
    }
]


class CameraSensor:

    def __init__(self, world, vehicle, camera):
        self.world = world
        self.name = camera['name']
        self.type = camera['type']
        bp = self.world.get_blueprint_library().find(self.type)

        if self.type.startswith('sensor.camera'):
            self.width = camera['width']
            self.height = camera['height']
            self.fov = camera['fov']
            bp.set_attribute("image_size_x", str(self.width))
            bp.set_attribute("image_size_y", str(self.height))  # set resolution
            bp.set_attribute("fov", str(self.fov))
            # 'iso': 100.0,
            # "exposure_mode": "histogram",
            # "exposure_compensation": -1.0,
            # "exposure_min_bright": 7.0,
            # "exposure_max_bright": 9.0,
            if self.type.endswith('rgb') and CARLA_VERSION == '0.9.12':
                bp.set_attribute('iso', str(camera['iso']))
                bp.set_attribute('exposure_mode', str(camera['exposure_mode']))
                bp.set_attribute('exposure_compensation', str(camera['exposure_compensation']))
                bp.set_attribute('exposure_min_bright', str(camera['exposure_min_bright']))
                bp.set_attribute('exposure_max_bright', str(camera['exposure_max_bright']))

        elif self.type.startswith('sensor.lidar'):
            bp.set_attribute('range', str(camera['range']))
            bp.set_attribute('channels', str(camera['channels']))
            bp.set_attribute('upper_fov', str(camera['upper_fov']))
            bp.set_attribute('lower_fov', str(camera['lower_fov']))
            bp.set_attribute('rotation_frequency', str(camera['rotation_frequency']))
            bp.set_attribute('points_per_second', str(camera['points_per_second']))
        else:
            assert False, 'camera type error'

        self.sensor = self.world.spawn_actor(bp, camera['pos'], attach_to=vehicle)
        self.convertor = camera['convertor']

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self, frame0):
        event = self.que.get()
        assert (event.frame == frame0)  # ensure synchronous

        if self.type.startswith('sensor.camera'):
            event.convert(self.convertor)
            img = np.array(event.raw_data)  # BGRA 32-bit pixels
            img = img.reshape((self.height, self.width, 4))[:, :, :3]  # BGR
            return img

        elif self.type.startswith('sensor.lidar'):
            points = np.frombuffer(event.raw_data, dtype=np.dtype('f4'))
            if CARLA_VERSION == '0.9.7':
                points = np.reshape(points, (int(points.shape[0] / 3), 3))
            elif CARLA_VERSION == '0.9.12':
                points = np.reshape(points, (int(points.shape[0] / 4), 4))
            else:
                raise NotImplementedError
            return points, event.transform

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class CollisionSensor:

    def __init__(self, world, vehicle, name):
        self.world = world
        self.name = name

        self.history = 0.0  # the intensity of the collision, 0.0 means pristine
        self.other_actor = None

        bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self, frame0):
        """
        Returns:
            self.hitory -- the max collision intensity
            self.other_actor -- the bp of the collided actor
        """

        if not self.que.empty():
            event = self.que.get()
            impulse = event.normal_impulse
            intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)  # calc the intensity

            if intensity > self.history:
                self.history = intensity
                self.other_actor = event.other_actor.type_id

        return (self.history, self.other_actor)

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()


class LaneInvasionSensor:

    def __init__(self, world, vehicle, name):
        self.world = world
        self.name = name

        bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = self.world.spawn_actor(bp, carla.Transform(), attach_to=vehicle)

        self.que = queue.Queue()
        self.sensor.listen(self.que.put)

    def get_data(self, frame0):
        """
        Returns:
            set() -- a set contains all the lane that invaded
        """

        if self.que.empty():
            return set()

        event = self.que.get()
        return set(x.type for x in event.crossed_lane_markings)

    def destroy(self):
        self.sensor.stop()
        self.sensor.destroy()