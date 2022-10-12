from pathlib import Path
from enum import Enum

"""
Configer is used to return the environment setting
"""
# CARLA_VERSION = '0.9.7'
CARLA_VERSION = '0.9.12'
if CARLA_VERSION == '0.9.7':
    from env.scenes_97 import *
    weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon']
    new_weathers = ['ClearSunset', 'CloudySunset', 'WetSunset', 'HardRainSunset']
    collect_weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon']
    weather_set = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon',
                   'ClearSunset', 'CloudySunset', 'WetSunset', 'HardRainSunset']

elif CARLA_VERSION == '0.9.12':
    from env.scenes_912 import *
    weathers = ['ClearNoon', 'CloudyNoon', 'MidRainyNoon', 'ClearSunset', 'CloudySunset', 'MidRainSunset']
    new_weathers = ['ClearNight', 'CloudyNight', 'MidRainyNight', 'WetNoon', 'WetSunset', 'WetNight']
    collect_weathers = ['ClearNoon', 'ClearSunset', 'ClearNight', 'MidRainyNoon', 'MidRainSunset', 'MidRainyNight',
                        'CloudyNoon', 'CloudySunset', 'CloudyNight', 'WetNoon', 'WetSunset', 'WetNight']
    weathers_prob = [1./6, 1./6, 1./6, 1./6, 1./6, 1./6]
    weather_set = ['ClearNight', 'ClearNoon', 'ClearSunset', 'CloudyNight', 'CloudyNoon',
                   'CloudySunset', 'Default', 'HardRainNight', 'HardRainNoon',
                   'HardRainSunset', 'MidRainSunset', 'MidRainyNight', 'MidRainyNoon',
                   'SoftRainNight', 'SoftRainNoon', 'SoftRainSunset', 'WetCloudyNight',
                   'WetCloudyNoon', 'WetCloudySunset', 'WetNight', 'WetNoon', 'WetSunset']
else:
    raise NotImplementedError

branch_list = [-1, 0, 1]

# direction of car when it starts
Direction = Enum('Direction', ('North', 'East', 'South', 'West'))
obs_shape_default = {'rgb': [3, 88, 200], 'depth': [3, 88, 200], 'lidar': [720], 'measurement': [7]}

# bird-view route
PATH_TO_ROUTES = Path('./data/path/')
PATH_TO_MAPS = Path('./data/maps')
width_in_pixels = 224
pixels_ev_to_bottom = 60
pixels_per_meter = 5.0

walkers_number_dict = {'few': [0, 10], 'moderate': [10, 20], 'standard': [20, 30], 'many': [35, 50]}

# obs_shape_default={'rgb': [3, 256, 256], 'sem': [3, 256, 256], 'depth': [3, 256, 256], 'lidar': [360], 'measurement': [7]}
def Configer(scene_id):
    return eval('scene' + str(scene_id) + '()')
