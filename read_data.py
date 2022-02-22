"""
An example script for read the dataset
"""
import os
import random
from collections import deque, OrderedDict

import h5py
import numpy as np


class Dataset:
    """ An example wrapper for loading the hdf5 dataset"""
    def __init__(self, size, branch_list, load_data):
        self.buffers = OrderedDict()
        self.branch_list = branch_list
        # keys in hdf5 file
        self.keys = ['img_t', 'lid_t', 'mea_t', 'com_t', 'loc_t', 'rot_t', 'a_t', 'pose', 'town',
                     'scene', 'trace', 'weather', 'done']

        for branch in branch_list:
            if size is None:
                self.buffers[branch] = deque()  # no length limitation
            else:
                self.buffers[branch] = deque(maxlen=size)

        if load_data != '':
            self.load_from_h5py(load_data)

    def add(self, branch, item):
        self.buffers[branch].append(item)

    def pop(self, branch):
        item = self.buffers[branch].pop()
        return item

    def count(self, branch):
        return len(self.buffers[branch])

    def get_batch(self, branch, batch_size):
        """return a dict(keys : img_t, lid_t, ...)"""
        samples = random.sample(self.buffers[branch], batch_size)
        return self.encoder(samples)

    def encoder(self, buffer):
        res = {key: [] for key in self.keys}
        for item in buffer:
            # item: img_t, lid_t, mea_t, com_t, loc_t, rot_t, a_t, pose, town, scene, trace, weather, done
            img_t, lid_t, mea_t, com_t, loc_t, rot_t, a_t, pose, town, scene, trace, weather, done = item
            res['img_t'].append(img_t)
            res['lid_t'].append(lid_t)
            res['mea_t'].append(mea_t)
            res['com_t'].append(com_t)
            res['loc_t'].append(loc_t)
            res['rot_t'].append(rot_t)
            res['a_t'].append(a_t)
            res['pose'].append(pose)
            res['town'].append(town)
            res['scene'].append(scene)
            res['trace'].append(trace)
            res['weather'].append(weather)
            res['done'].append(done)
        for key in self.keys:
            res[key] = np.array(res[key])
        return res

    def save_to_h5py(self, filename):
        with h5py.File(filename, 'w') as f:
            for branch in self.branch_list:
                g = f.create_group('branch%d' % branch)
                res = self.encoder(self.buffers[branch])
                for k, v in res.items():
                    g.create_dataset(k, data=v)
        print('# Save Data successfully!')
        for branch in self.branch_list:
            print('Branch= %d, Count= %d' % (branch, self.count(branch)))

    def decoder(self, buffer, dataset):
        res = {key: dataset[key][:] for key in self.keys}
        cnt = len(res['img_t'])
        assert all(cnt == len(res[key]) for key in self.keys)
        for i in range(cnt):
            buffer.append((res['img_t'][i],
                           res['lid_t'][i],
                           res['mea_t'][i],
                           res['com_t'][i],
                           res['loc_t'][i],
                           res['rot_t'][i],
                           res['a_t'][i],
                           res['pose'][i],
                           res['town'][i],
                           res['scene'][i],
                           res['trace'][i],
                           res['weather'][i],
                           res['done'][i]
                           ))

    def load_from_h5py(self, filename):
        with h5py.File(filename, 'r') as f:
            for branch in self.branch_list:
                self.decoder(self.buffers[branch], f['branch%d' % branch])
        print('# load data from %s successfully!' % filename)
        for branch in self.branch_list:
            print('Branch= %d, Count= %d' % (branch, self.count(branch)))

    def statistics(self):
        # frames/trajectories by scene
        s_frame_cnt = {}
        s_traj_cnt = {}
        # frames/trajectories by mission
        b_frame_cnt = {branch: 0 for branch in self.branch_list}
        b_traj_cnt = {branch: set() for branch in self.branch_list}
        # frames by Lat. Cmd & Lon. Cmd
        lat_frame_cnt = {}
        lon_frame_cnt = {}
        for branch in self.branch_list:
            for item in self.buffers[branch]:
                scene = item[9]
                if not scene in s_frame_cnt.keys():
                    s_frame_cnt[scene] = 0
                s_frame_cnt[scene] += 1
                if not scene in s_traj_cnt.keys():
                    s_traj_cnt[scene] = set()
                s_traj_cnt[scene].add(item[10])

                b_frame_cnt[branch] += 1
                b_traj_cnt[branch].add(item[10])

                lat_cmd, lon_cmd = item[3]
                if not lat_cmd in lat_frame_cnt.keys():
                    lat_frame_cnt[lat_cmd] = 0
                lat_frame_cnt[lat_cmd] += 1
                if not lon_cmd in lon_frame_cnt.keys():
                    lon_frame_cnt[lon_cmd] = 0
                lon_frame_cnt[lon_cmd] += 1

        for s in sorted(s_frame_cnt.keys()):
            print("Scene %d, frames %d, trajectories %d" % (s, s_frame_cnt[s], len(s_traj_cnt[s])))
        for b in sorted(b_frame_cnt.keys()):
            print("Branch %d, frames %d, trajectories %d" % (b, b_frame_cnt[b], len(b_traj_cnt[b])))
        for lat_cmd in sorted(lat_frame_cnt.keys()):
            print("Lat.Cmd %d, frames %d" % (lat_cmd, lat_frame_cnt[lat_cmd]))
        for lon_cmd in sorted(lon_frame_cnt.keys()):
            print("Lon.Cmd %d, frames %d" % (lon_cmd, lon_frame_cnt[lon_cmd]))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--branch_list', nargs='+', type=int, default=[-1, 0, 1],
                        help='branch list (left turn|straight|right turn) (-1|0|1)')
    parser.add_argument('--load_data', type=str, required=True)
    args = parser.parse_args()
    data = Dataset(100000, args.branch_list, args.load_data)
    data.get_batch(random.choice(args.branch_list), 100)
    data.statistics()

