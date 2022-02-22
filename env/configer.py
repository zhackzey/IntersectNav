from enum import Enum

"""
Configer is used to return the environment setting
"""

branch_list = [-1, 0, 1]
# branch_list = [-1]
weathers = ['ClearNoon', 'CloudyNoon', 'WetNoon', 'HardRainNoon']
new_weathers = ['ClearSunset', 'CloudySunset', 'WetSunset', 'HardRainSunset']
# direction of car when it starts
Direction = Enum('Direction', ('North', 'East', 'South', 'West'))


def Configer(scene_id):
    return eval('scene' + str(scene_id) + '()')


#############################
#   Scene Configuration
############################

class scene:

    def __init__(self, ped_center, ped_range, area, NumOfWal, NumOfVeh, poses, ends, direction, lane_type, town,
                 scene_id):
        self.ped_center = ped_center
        self.ped_range = ped_range
        self.area = area  # minX  maxX  minY maxY

        self.NumOfWal = NumOfWal
        self.NumOfVeh = NumOfVeh

        self.poses = poses
        self.ends = ends
        self.direction = direction
        task_types = {Direction.North: [-1, 1, 0, 2], Direction.South: [1, -1, 2, 0],
                      Direction.West: [0, 2, 1, -1], Direction.East: [2, 0, -1, 1]}
        self.task_type = [task_types[i] for i in self.direction]
        self.lane_type = lane_type

        self.town = town
        self.scene_id = scene_id

    def poses_num(self):
        return len(self.poses)

    def branches(self, pose):
        assert pose in set(range(self.poses_num()))

        branches = []
        for b, v in zip(branch_list, self.lane_type[pose]):
            if v == 1:
                branches.append(b)

        return branches

    #    def get_trace_id(self, pose, branch):
    #        trace = 0
    #        for i in range(pose+1):
    #            for j in range(len(branch_list)):
    #                if (self.lane_type[i][j] == 1) & (i < pose | j < branch_list.index(branch)):
    #                    trace = trace + 1
    #        return trace

    def scene_config(self, pose, branch):
        assert pose in set(range(self.poses_num()))
        assert branch in self.branches(pose), "not consistent with lane type"
        #       trace = self.get_trace_id(pose, branch)
        scene = {
            'ped_center': self.ped_center,
            'ped_range': self.ped_range,
            'area': self.area,

            'Wait_ticks': 50,
            'weather': 'ClearNoon',

            'NumOfWal': self.NumOfWal,
            'NumOfVeh': self.NumOfVeh,

            'start': self.poses[pose],
            'end': self.ends[pose][branch_list.index(branch)],
            'task_type': self.task_type[pose],
            'lane_type': self.lane_type[pose],
            'branch': branch,
            'pose': pose,

            'town': self.town,
            'scene_id': self.scene_id,
            #            'trace': trace,
        }
        return scene


class scene0(scene):

    def __init__(self):
        super().__init__(
            ped_center=741,
            ped_range=30.0,
            area=[-102.0, -60.0, 115.0, 152.0],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[1760, 1917, 1883],
            ends=[[-1, 1844, 1367], [2327, 2001, -1], [-1, 544, 2327]],
            direction=[Direction.North, Direction.North, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [0, 1, 1]],

            town='Town03',
            scene_id=0,
        )


class scene1(scene):

    def __init__(self):
        super().__init__(
            ped_center=610,
            ped_range=30.0,
            area=[-27.0, 25.0, 111.0, 153.0],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[1219, 593, 2015],
            ends=[[-1, 2719, 1234], [1997, 79, -1], [1234, 52, -1]],
            direction=[Direction.North, Direction.North, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [1, 1, 0]],

            town='Town03',
            scene_id=1,
        )


class scene2(scene):

    def __init__(self):
        super().__init__(
            ped_center=1374,
            ped_range=30.0,
            area=[-18.0, 25.0, -155.0, -117.0],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[2737, 1927, 1725, 2204],
            ends=[[-1, 801, 353], [95, 1307, -1], [353, 346, -1], [-1, 847, 95]],
            direction=[Direction.North, Direction.North, Direction.South, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]],

            town='Town03',
            scene_id=2,
        )


class scene3(scene):

    def __init__(self):
        super().__init__(
            ped_center=2200,
            ped_range=30.0,
            area=[-102, -57, -160, -110],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[368, 1323, 933],
            ends=[[-1, 806, 417], [2335, 2463, -1], [417, 1637, -1]],
            direction=[Direction.North, Direction.North, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [0, 1, 0]],

            town='Town03',
            scene_id=3,
        )


class scene4(scene):

    def __init__(self):
        super().__init__(
            ped_center=361,
            ped_range=30.0,
            area=[10, 45, -35, 35],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[3762, 3758, 3664, 3660],
            ends=[[-1, 3632, 555], [2288, 3622, -1], [544, 3736, -1], [-1, 3747, 297]],
            direction=[Direction.North, Direction.North, Direction.South, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]],

            town='Town05',
            scene_id=4,
        )


class scene5(scene):

    def __init__(self):
        super().__init__(
            ped_center=305,
            ped_range=30.0,
            area=[-70, -20, -30, 30],

            NumOfWal=[20, 30],
            NumOfVeh=[0, 0],

            poses=[4559, 4554, 4533, 4530],
            ends=[[-1, 4692, 2310], [2234, 4681, -1], [2299, 4415, -1], [-1, 4406, 2223]],
            direction=[Direction.North, Direction.North, Direction.South, Direction.South],
            lane_type=[[0, 1, 1], [1, 1, 0], [1, 1, 0], [0, 1, 1]],

            town='Town05',
            scene_id=5,
        )
