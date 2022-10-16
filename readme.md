# IntersectNav Benchmark

Repository to store the IntersectNav Benchmark for CARLA driving simulator. The agent needs to navigate through dense intersections and interact with pedestrians at crosswalks.
### Refer to the [video](http://www.poss.pku.edu.cn/OpenDataResource/IntersectionNav2022/video.mp4) associated with our paper for more information.
### Here is an alternative [URL](https://drive.google.com/file/d/12G1VahVuxHutKXhnulms2aMsaeMUe1I4/view?usp=sharing). 
## Requirements

### Installation of CARLA simulator

CARLA simulator is in active development with many versions. Click here for its [official repository]( https://github.com/carla-simulator/carla) . 

In our experiments, the interaction with only pedestrians scenarios is simulated in CARLA 0.9.7, while interaction with both pedestrians and environmental vehicles is simulated in CARLA 0.9.12.

**This benchmark supports two versions or CARLA. Although any versions above 0.9.7 is compatible with our benchmark in principle, we suggest use of version 0.9.7 and 0.9.12 for consistence.**

A detailed installation of CARLA simulator 0.9.7 and 0.9.12 is available at the official documentation [CARLA 0.9.7](https://carla.readthedocs.io/en/0.9.7/how_to_build_on_linux/) and [CARLA 0.9.12](https://carla.readthedocs.io/en/0.9.12/start_quickstart/).
For any issues occurred during the installation, please refer to the official guidelines and F.A.Qs.

### Installation of python packages

The full requirements list is available at *environment.yml*.

- numpy
- scipy
- ffmpeg
- opencv
- carla=0.9.7
- ... 

We suggest use of Anaconda. For quick installation of the python packages, run

```shell
conda env create -n carla97 -f environment.yml
conda activate carla97
```

## Benchmark Configuration

Our benchmark can be easily configured in python script [configer.py](env/configer.py), [scenes_97.py](env/scenes_97.py), [scenes_912.py](env/scenes_912.py)

#### Tasks

For now, we consider three tasks of navigation through the intersections (left turn, -1 |go straight, 0|right turn, 1).

#### Weathers

The train weathers contain (ClearNoon, CloudyNoon, WetNoon, HardRainNoon)

The test weathers contain (ClearSunset, CloudySunset, WetSunset, HardRainSunset)

#### Scenes

![scene_illustration](figs/scene_illustration.png)

Currently our benchmark inludes six intersections/scenes at two towns in CARLA simulator. Each scene is configured by a series of components (e.g. its intersection center, size, weather, number of pedestrians, start/end locations, etc.).

An example is as follows:

```python
class scene0(scene):
    def __init__(self):
        super().__init__(
            ped_center=741, # index of the intersection center waypoint
            ped_range=30.0, # intersection area size
            area=[-102.0, -60.0, 115.0, 152.0], # area in coordinates

            NumOfWal=[20, 30], # the range of number of walkers
            NumOfVeh=[0, 0], # the range of number of vehicles

            poses=[1760, 1917, 1883], # indexes of start waypoints
            ends=[[-1, 1844, 1367], [2327, 2001, -1], [-1, 544, 2327]], # indexes of end waypoints
            direction=[Direction.North, Direction.North, Direction.South], # directions
            lane_type=[[0, 1, 1], [1, 1, 0], [0, 1, 1]], # lane types

            town='Town03', # town index in CARLA
            scene_id=0, # scene index
        )
```

New intersections can be added by determining the appropriate configuration as above.

#### Sensors

On-board sensors can be configured in [sensors.py](env/sensors.py).

For now, following sensors are available.

- Front-view RGB camera, $W \times H = 1500 \times 650$, fov = 140.
- Ray-cast LiDAR, range 40m, one channel.
- CollisionSensor, based on CARLA blueprint *sensor.other.collision*
- LaneInvasionSensor, based on CARLA blueprint *sensor.other.lane_invasion*

## Protocol & Evaluation

For each intersection, we configure the ego vehicle's available initial locations and corresponding goal locations, which contain different lanes and directions around the intersection. The initial heading is always towards the intersection. The benchmark adopts an episodic setup. At each episode, one of the intersections is selected and the ego car randomly starts from one of the available configurations. Three tasks, i.e., performing left turn/go straight/right turn and navigate through the intersection need to be completed. Meanwhile, a random number of 20-30 pedestrians are generated to walk through the crosswalks  around intersections. Apart from pedestrians, the simulation environment can be configured to introduce environmental vehicles. In each episode, a random number of 6-15 environmental vehicles are generated at random spawn points around the intersection area, which are controlled by CARLA built-in autopilot agents. All available routes for these intersections add up to about 40.

During the close-loop simulation for evaluation, the ego agent, pedestrians and environmental vehicles (optional) initialized according to above protocols. 

 Then the network predicted values are clipped by the range $[-1.0, 1.0]$ and passed to the actuator in CARLA simulation, which simulates the world's dynamics and move on to the next step. This process iterates until an episode is done. We record the outcomes and metrics as follows.

### Evaluation Metrics

The metrics are made to evaluate the driving performance for different considerations. We define the possible outcomes for one episode as follows:

- **Timeout**: Failure to arrive at the goal point within limited time steps (set to 1000).
- **Lane Invasion**: The number of illegal lane invasions (e.g. drift onto the opposite lanes) exceeds 5.
- **Collision**:Once the agent collides into any object (pedestrians or obstacles), it is considered as a collision.
- **Success**: We only consider one episode as a success when the agent arrives at the goal without any of the above conditions triggered. 

The ratios of above outcomes are calculated. For example:

$success\ rate = \frac{number\ of\ success\ rollouts}{number\ of\ all\ rollouts}$

There are continuous metrics to evaluate the model's control quality. Detailed information can be found in the paper.

- Ego Jerk
- Other Jerk
- Deviation from waypoint
- Deviaion from destination
- Heading angle deviation
- Total steps

### Run Evaluation Script

We provide a **gym-like** environment interface named **Env** in [carla97_env.py](env/carla97_env.py) .

We also provide a basic agent that uses A* search for global route planning and PID for control. 

The standard evaluate iteration is as follows:

```python
env = Env(port=2000)
s_t, info_t = env.reset(scene)
for i in range(max_iter):
    # retrieve input data & high-level commands from s_t
    img_t = s_t[0]
    mea_t = s_t[2]
    cmd_t = s_t[3]
    # predict actions
    a_t = model.predict(img_t, mea_t, cmd_t)
    # world step, return the next state and information
    # control agent specifies the type (PID|PID_NOISE|NN)
    s_t1, r_t, done, info_t, _ = self.env.step(a_t, lateral=control_agent, longitude=control_agent)
    s_t = s_t1
```

We provide an evaluation script [evaluate.py](evaluate.py).

Available command line arguments are:

```shell
# usage: evaluate.py [-h] [--work_dir WORK_DIR] [--model_path MODEL_PATH]
#                   --branch_list BRANCH_LIST [BRANCH_LIST ...] --scenes SCENES
#                   [SCENES ...] [--control_agent {NN,PID,PID_NOISE}]
#                   [--iters ITERS] [--eval_steps EVAL_STEPS] [--eval_render]
#                   [--store_success_only] [--eval_save EVAL_SAVE]
#                   [--port PORT] [--random_weather] [--new_weather]
#                   [--gpu_id GPU_ID] [--breakpoint BREAKPOINT]
#
# Evaluate one agent's performance in IntersectNav benchmark
#
# optional arguments:
#  -h, --help            show this help message and exit
#  --work_dir WORK_DIR   working directory
#  --model_path MODEL_PATH
#                        NN checkpoint path
#  --branch_list BRANCH_LIST [BRANCH_LIST ...]
#                        evaluating task branches, (left turn|straight|right
#                        turn) (-1|0|1)
#  --scenes SCENES [SCENES ...]
#                        scene IDs for evaluating
#  --control_agent {NN,PID,PID_NOISE}
#                        agent options for control
#  --iters ITERS         evaluate episodes
#  --eval_steps EVAL_STEPS
#                        maximum env steps
#  --eval_render         render image
#  --store_success_only  only store data of successful episodes
#  --eval_save EVAL_SAVE
#                        save (s_t, a_t, ...) pairs per certain interval
#  --port PORT           carla host port
#  --random_weather      random weather
#  --new_weather         use new weathers
#  --gpu_id GPU_ID       gpu ID
#  --breakpoint BREAKPOINT
#                        evaluation breakpoint for continue
```

Example usages are listed below:

```shell
# evaluate a PID agent on scene 0 under random test weathers, tasks (left|straight|right)
python evaluate.py --scenes 0 --control_agent PID --eval_save 4 --new_weather --branch_list -1 0 1 --random_weather
# evaluate a NN agent on scene 0 & 1 under random train weathers, each route 5 episodes, tasks (left|straight)
python evaluate.py --scenes 0 1 --control_agent NN --model_path path_to_model_ckpt --eval_save 2 --branch_list -1 0 --random_weather --ites 5
```

## Human Demonstration Data

We collected two human driving datasets named **Ped-Only** and **Ped-Veh** under this benchmark's scenarios. For more details, please refer to our paper.

**Ped-Only**: over 950 trajectories on 6 scenes. Only pedestrians are present.

**Ped-Veh**: over 300 trajectories on 2 scenes. Both pedestrians and environmental vehicles are present.

[The dataset can be downloaded here](http://www.poss.pku.edu.cn/OpenDataResource/IntersectionNav2022/IntersectionNavHumanDataset.zip) 

### Dataset **Ped-Only**
The data is stored on two HDF5 files. 

- PedOnly_scene012345_train.h5: dataset for train
- PedOnly_scene012345_test.h5: dataset for validation and test

### Dataset **Ped-Veh**
The data is stored in file PedVeh_scenes01.h5.

Each HDF5 contains three groups:

1. branch -1: data belonging to a left-turn task
2. branch 0: data belonging to a go-straight task
3. branch 1: data belonging to a right-turn task

Each branch data contains the following fields:

1. 'img_t': front-view RGB images, shape (88, 200, 3), uint8
2. 'lid_t': lidar range, shape (720,), float
3. 'mea_t': measuremets, shape (3,), float
   - current speed, float
   - distance to nearest waypoint, float
   - angle deviation, the deviation between current heading and the goal point direction, float
4. 'com_t': high-level commands, shape (2,), int
   - lateral command: 0 follow lane, 1 turn left, 2 turn right, 3 go straight
   - longitude command: 0 decelerate, 1 maintain, 2 accelerate
5. 'loc_t': location coordinates (x,y,z), shape (3,), float
6. 'rot_t': rotation, degrees (pitch, yaw, roll), shape (3,), float
7. 'a_t': control actions, shape (3,), float
   - steer, float
   - throttle, float
   - brake, float
8. 'pose': route pose/start waypoint, int
9. 'town': town index, int
10. 'scene': scene index, int
11. 'trace': trace index, int
12. 'weather': weather index, int
13. 'done': indicator, whether this episode is terminated, boolean

### Data parse

We provide an example  script [read_data.py](read_data.py). For detailed information, please refer to the code. Options can be found by

```shell
python read_data.py --help
```

## Paper & Citation

If you use our benchmark, please cite our paper below.

Zeyu, Zhu and Huijing, Zhao. [[PDF](https://arxiv.org/pdf/2202.10124)]
```
@misc{zhu2022multitask,
      title={Multi-Task Conditional Imitation Learning for Autonomous Navigation at Crowded Intersections}, 
      author={Zeyu Zhu and Huijing Zhao},
      year={2022},
      eprint={2202.10124},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

 