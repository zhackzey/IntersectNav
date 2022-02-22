"""
    Evaluation protocol, which evaluates one agent's performance in IntersectNav benchmark
"""
import argparse
import math as m
import os
import random
import time
from collections import OrderedDict
from datetime import datetime

import csv
import cv2
import numpy as np
import torch
from tqdm import trange

from env.carla97_env import Env
from env.configer import Configer, weathers, new_weathers
from read_data import Dataset


def render_image(name, img):
    """using cv2

    Arguments:
        name {str}
        img {np.array}
    """
    cv2.namedWindow(name)
    cv2.imshow(name, img)
    cv2.waitKey(3)


def video_write(path, frames, size=(1500, 650), fps=30):
    videoWriter = cv2.VideoWriter(path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
    for frame in frames:
        videoWriter.write(frame)
        cv2.waitKey(1)
    videoWriter.release()


def create_csv(path, csv_head):

    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(csv_head)


def write_csv(path, data_row):

    with open(path,'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)


class Evaluator:
    def __init__(self, args, model, device):
        self.branch_list = args.branch_list
        self.iters = args.iters
        self.eval_steps = args.eval_steps
        self.eval_render = args.eval_render
        self.store_success_only = args.store_success_only
        self.eval_save = args.eval_save
        self.port = args.port
        self.random_weather = args.random_weather

        self.control_agent = args.control_agent
        self.work_dir = args.work_dir
        self.model = model

        self.env = Env(args.port)
        self.buffers = Dataset(int(1e10), args.branch_list, '')
        self.fp = os.path.dirname(__file__)
        self.breakpoint_path = args.breakpoint
        self.trace_id = 0
        self.device = device
        self.new_weather = args.new_weather
        # train weathers or test weathers?
        self.weather_set = weathers if not self.new_weather else new_weathers
        if not self.random_weather:
            self.weather_set = ['ClearNoon']
        ts = datetime.now().strftime("%m_%d_%H_%M")
        csv_dir_name = 'eval_scenes-%s_branches-%s_%s-weather_control-%s_%s' % \
                       (''.join(map(str, args.scenes)),
                        ''.join(map(str, args.branch_list)),
                        'train' if not args.new_weather else 'test',
                        self.control_agent,
                        ts)
        self.csv_dir = os.path.join(self.work_dir, csv_dir_name)
        # create directory
        if not os.path.exists(self.csv_dir):
            os.mkdir(self.csv_dir)
        self.lat_cmd2text = {0: 'Follow Lane', 1: 'Turn Left', 2: 'Turn Right', 3: 'Go Straight'}
        self.lat_cmd2color = {0: (255, 255, 0), 1: (0, 0, 255), 2: (0, 255, 0), 3: (255, 0, 0)}
        self.lon_cmd2text = {0: 'Decelerate', 1: 'Maintain', 2: 'Accelerate'}
        self.lon_cmd2color = {0: (255, 0, 0), 1: (255, 255, 255), 2: (0, 255, 0)}
        if self.control_agent == 'NN':
            assert model, "No NN model is provided!"
        fn = 'Control_%s' % self.control_agent
        self.csv_path = os.path.join(self.csv_dir, fn + time.strftime('_%m%d_%H%M') + '.csv')
        self.txt_path = os.path.join(self.csv_dir, fn + time.strftime('_%m%d_%H%M') + '.txt')
        self.save_data = os.path.join(self.csv_dir, 'data' + time.strftime('_%m%d_%H%M') + '.h5')
        print("[INFO]: save data path: ", self.save_data)
        self.first_write = True
        self.txt_created = False
        if self.breakpoint_path:
            if self.breakpoint_path.startswith('eval'):
                pass
            else:
                self.breakpoint_path = os.path.join(self.csv_dir, self.breakpoint_path)
        if not os.path.exists(self.breakpoint_path):
            self.breakpoint_path = ''
        else:
            print("[INFO]: breakpoint file: ", self.breakpoint_path)

    def evaluate_episode(self, scene, pose):
        if scene['town'] != self.env.world.get_map().name:
            self.env = Env(self.port, town=scene['town'])
        if self.random_weather:
            scene['weather'] = random.choice(self.weather_set)
        s_t, info_t = self.env.reset(scene)
        video = []
        actions = []
        lat_cmds = []
        lat_cmds_colors = []
        lon_cmds = []
        lon_cmds_colors = []
        tmp_buffer = []
        for t in trange(self.eval_steps, desc="step", unit="step"):
            branch = scene['branch']
            # rgb, shape: (88, 200, 3)
            # mea, shape: (4,)
            rgb, lid, mea, command, location, rotation = s_t
            lat_command, lon_command = command
            lat_cmds.append(lat_command)
            lon_cmds.append(lon_command)
            lat_cmds_colors.append('#%02x%02x%02x' % (
                self.lat_cmd2color[lat_command][2],
                self.lat_cmd2color[lat_command][1],
                self.lat_cmd2color[lat_command][0]))
            lon_cmds_colors.append('#%02x%02x%02x' % (
                self.lon_cmd2color[lon_command][2],
                self.lon_cmd2color[lon_command][1],
                self.lon_cmd2color[lon_command][0]))
            if self.control_agent == 'NN':
                # transpose to PyTorch CxHxW and normalize
                img = np.transpose(rgb, (2, 0, 1)) / 255.0
                # use speed only?
                # mea = mea[0]
                a_t = np.zeros([2])
                ''' 
                # complete the following code 
                # neural network inference to predict a_t given img, mea and command
                # a_t = self.model.predict(img, mea, command)
                '''
            else:
                a_t = np.zeros([2])
            actions.append(a_t)
            try:
                s_t1, r_t, done, info_t, _ = self.env.step(a_t,
                                                           lateral=self.control_agent,
                                                           longitude=self.control_agent)
            except:
                # something nasty happened :(
                print("[ERROR] Error happened: step the world")
                return None, None, None
            else:
                pass

            if self.eval_save > 0 and t % self.eval_save == 0:
                tmp_buffer.append((rgb, lid, mea, command, location, rotation, info_t['a_t'], pose,
                                   int(scene['town'][4:]), scene['scene_id'], self.trace_id,
                                   self.weather_set.index(scene['weather']), [done]))

            if self.eval_render:
                render_image('FrontRGB', info_t['FrontRGB'])
            modified_photo = np.array(info_t['FrontRGB'])
            cv2.putText(modified_photo, 'steer: {}'.format(round(info_t['a_t'][0], 3)), (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 1)
            cv2.putText(modified_photo, 'throttle-brake: {}'.format(round(info_t['a_t'][1] - info_t['a_t'][2], 3)), (220, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(modified_photo, 'lateral command: {}'.format(self.lat_cmd2text[lat_command]),
                        (220, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.lat_cmd2color[lat_command], 1)
            cv2.putText(modified_photo, 'longitude command {}'.format(self.lon_cmd2text[lon_command]),
                        (220, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.lon_cmd2color[lon_command], 1)

            video.append(modified_photo)
            s_t = s_t1
            if done:
                break

        # judgement of angle and distance
        file_path = './destination/scene%d_branch%d_pose%d.txt' % (
            scene['scene_id'], scene['branch'], pose)
        assert os.path.exists(file_path), "Destination file does exist on disk !!"

        with open(file_path) as f:
            tmp = f.readline().split(' ')
            expert_location = list(map(float, tmp[:2]))
            expert_yaw = float(tmp[-1])

        self.env.res['dis_to_destination'] = m.sqrt((expert_location[0] - info_t['location'].x) ** 2 +
                                                    (expert_location[1] - info_t['location'].y) ** 2)
        self.env.res['vertical_dist'] = abs(
            -m.sin(m.pi / 180 * expert_yaw) * (expert_location[0] - info_t['location'].x) +
            m.cos(m.pi / 180 * expert_yaw) * (expert_location[1] - info_t['location'].y))
        self.env.res['delta_angle'] = (info_t['rotation'].yaw - expert_yaw + 180) % 360 - 180
        self.env.res['tot_step'] = self.env.frame - self.env.start_frame
        if self.env.res['tot_step'] == self.eval_steps:
            self.env.res['time_out'] = True

        if not self.env.res['collision'] and not self.env.res['lane_invasion'] and not self.env.res['time_out']:
            # done is not due to collision/invasion/timeout
            if self.env.res['wrong_direction']:
                # drive to the wrong direction
                self.env.res['success'] = False
            else:
                if self.env.res['vertical_dist'] > 1:
                    self.env.res['TooFar'] = True
                if abs(self.env.res['delta_angle']) > 15:
                    self.env.res['TooMuchAngle'] = True
                # NOTE: TooFar and TooMuchAngle can happen at the same time
                if self.env.res['TooFar'] or self.env.res['TooMuchAngle']:
                    self.env.res['success'] = False
        poor_dis_or_angle = 1.0 if self.env.res['TooMuchAngle'] or self.env.res['TooFar'] else 0.0
        success, collision, lane_invasion, time_out = float(self.env.res['success']), \
                                                      float(self.env.res['collision']), \
                                                      float(self.env.res['lane_invasion']), \
                                                      float(self.env.res['time_out'])
        if self.eval_save > 0:
            if self.store_success_only and not self.env.res['success']:
                pass
            else:
                for item in tmp_buffer:
                    self.buffers.add(scene['branch'], item)
                self.buffers.save_to_h5py(self.save_data)
        return self.env.res, self.env.close(self.eval_steps), video

    def evaluate_scene(self, scene_list):
        episode_bk, scene_bk, pose_bk, branch_bk, trace_bk = [-1, -1, -1, -1, -1]
        if self.breakpoint_path:
            with open(self.breakpoint_path) as f:
                episode_bk, scene_bk, pose_bk, branch_bk, trace_bk = f.read().split()
                episode_bk = int(episode_bk)
                scene_bk = int(scene_bk)
                pose_bk = int(pose_bk)
                branch_bk = int(branch_bk)
                trace_bk = int(trace_bk)
            self.trace_id = trace_bk

        for scene_idx, scene_id in enumerate(scene_list):
            if scene_idx < scene_bk: continue
            configer = Configer(scene_id)
            for pose in range(configer.poses_num()):
                if scene_idx == scene_bk and pose < pose_bk: continue
                for branch_idx, branch in enumerate(configer.branches(pose)):
                    if branch not in self.branch_list: continue
                    if scene_idx == scene_bk and pose == pose_bk and branch_idx < branch_bk: continue
                    scene = configer.scene_config(pose, branch)
                    if self.random_weather:
                        scene['weather'] = random.choice(self.weather_set)
                    res_list = OrderedDict()
                    for episode in range(self.iters):
                        if scene_idx == scene_bk and pose == pose_bk and branch_idx == branch_bk and episode < episode_bk: continue
                        print('# Evaluate scene_id = %d, pose = %d, branch = %d, Episode = %d / %d, Trace_id = %d' % (
                            scene_id, pose, branch, episode, self.iters, self.trace_id))
                        res, _, video = self.evaluate_episode(scene, pose)
                        self.trace_id = self.trace_id + 1
                        if res is None:
                            continue
                        print(res)
                        video_name = 'eval_scene%d_pose%d_branch%d_episode%d' % (scene_id, pose, branch, episode) + \
                                     ('_suc' if res['success'] else '') + ('_timeout' if res['time_out'] else '') + \
                                     ('_laneInv' if res['lane_invasion'] else '') + \
                                     ('_colli' if res['collision'] else '') + \
                                     ('_poorDisAngle' if res['TooFar'] or res['TooMuchAngle'] else '') + \
                                     ('_wrongDirection' if res['wrong_direction'] else '') + '.avi'
                        print("video length:", len(video))
                        video_write(os.path.join(self.csv_dir, video_name), video, size=(1500, 650), fps=15)

                        for (k, v) in res.items():
                            if not k in res_list:
                                res_list[k] = []
                            res_list[k].append(v)

                        with open(self.txt_path, 'a') as f:
                            if not self.txt_created:
                                self.txt_created = True
                                f.write('scene pose branch ')
                                for (k, v) in res.items():
                                    f.write(k)
                                    f.write(' ')
                                f.write('\n')
                            f.write('%d %d %d ' % (scene_id, pose, branch))
                            for (k, v) in res.items():
                                f.write(str(v))
                                f.write(' ')
                            f.write('\n')
                            f.close()
                        with open(os.path.join(self.csv_dir, 'breakpoint.txt'), 'w') as f:
                            f.write("{} {} {} {} {}".format(episode, scene_idx, pose, branch_idx, self.trace_id))
                            f.close()

                    self.update_metrics(scene_id, pose, branch, res_list)

    def update_metrics(self, scene, pose, branch, res_list):
        res_mean = OrderedDict([('scene', scene), ('pose', pose), ('branch', branch)])
        for (k, v) in res_list.items():
            res_mean[k] = np.mean(v)

        if self.first_write:
            create_csv(self.csv_path, list(res_mean.keys()))
            self.first_write = False

        write_csv(self.csv_path, list(res_mean.values()))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one agent's performance in IntersectNav benchmark")
    # --------------------------- Model related ----------------
    parser.add_argument('--work_dir', default='.', type=str, help="working directory")
    parser.add_argument('--model_path', default='', type=str, help='NN checkpoint path')
    # --------------------------- Evaluation Configuration ----------------
    parser.add_argument('--branch_list', nargs='+', type=int, required=True,
                        help='evaluating task branches, (left turn|straight|right turn) (-1|0|1)')
    parser.add_argument('--scenes', nargs='+', type=int, required=True, help='scene IDs for evaluating')
    parser.add_argument('--control_agent', type=str, default='NN', choices=['NN', 'PID', 'PID_NOISE'],
                        help="agent options for control")
    parser.add_argument('--iters', default=1, type=int, help='evaluate episodes')
    parser.add_argument('--eval_steps', default=1000, type=int, help='maximum env steps')
    parser.add_argument('--eval_render', default=False, action='store_true', help='render image')
    parser.add_argument('--store_success_only', default=False, action='store_true',
                        help='only store data of successful episodes')
    parser.add_argument('--eval_save', default=-1, type=int, help='save (s_t, a_t, ...) pairs per certain interval')
    # --------------------------- CARLA simulator configuration ----------------
    parser.add_argument('--port', default=2000, type=int, help='carla host port')
    parser.add_argument('--random_weather', default=False, action="store_true", help="random weather")
    parser.add_argument('--new_weather', default=False, action="store_true", help="use new weathers")
    # --------------------------- Optional ---------------------------
    parser.add_argument('--gpu_id', default='0', type=str, help='gpu ID')
    parser.add_argument('--breakpoint', default='', type=str, help='evaluation breakpoint for continue')

    args = parser.parse_args()
    assert os.path.exists(args.work_dir), "work directory %s does not exist!" % args.work_dir
    if args.model_path:
        assert os.path.exists(args.model_path), "model %s does not exist!" % args.model_path
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if torch.cuda.is_available():
        print("=================== Using GPU, device count: {} =================".format(torch.cuda.device_count()))
        device = torch.device('cuda')
    else:
        print("=================== Using CPU =======================")
        device = torch.device('cpu')

    model = None
    '''
    # complete the following example code if u wants to evaluate neural network agent
    # e.g.
    # import my_model
    # model = my_model(*args, **argvs).to(device)
    # load from checkpoint
    # model.load_state_dict(args.model_path)
    # model.eval()
    '''

    print('[INFO]: Start evaluation on scenes:', args.scenes, ' and branches: ', args.branch_list)
    evaluator = Evaluator(args, model, device)
    evaluator.evaluate_scene(scene_list=args.scenes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\nExit by user.')
    finally:
        print('\nExit.')
