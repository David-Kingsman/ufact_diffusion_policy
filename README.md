# UFACTORY Diffusion Policy

**Training and Evaluation Experiments for Ufactory Xarm Robots**

## 1. Installation
### 1.1. Environement configuration
We recommend [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) instead of the standard anaconda distribution for faster installation: 

```console
$ mamba env create -f conda_environment_real.yaml
```

but you can use conda as well: 
```console
$ conda env create -f conda_environment_real.yaml
```

### 1.2. xArm Python SDK

Download [XArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK/tree/a54b2fd1922d3245f1f78c8e518871d4760ead1c)

```bash
mkdir third_party
cd third_party
git clone https://github.com/xArm-Developer/xArm-Python-SDK.git
cd xArm-Python-SDK
Install from pypi
```

Install from pypi from [XArm Python SDK](https://github.com/xArm-Developer/xArm-Python-SDK/tree/a54b2fd1922d3245f1f78c8e518871d4760ead1c)

```bash
pip install xarm-python-sdk
```

## 2. Data format

Collected data is stored in the [`demos_collected`](./demos_collected), with each run in a separate subfolder named `runXXX`, where `XXX` is the run number. Each run folder contains:

### 2.1. Folder Structure from the ufactory_teleoperation

```yaml
demos_collected/
├── run001/
│   ├── config.json                   # Configuration file
│   ├── demo_action.npz               # Action sequence data
│   ├── demo_ee_states.npz            # End effector states
│   ├── demo_target_pose_mat.npz      # Target pose matrix
│   ├── demo_joint_states.npz         # Joint states
│   ├── demo_gripper_states.npz       # Gripper states
│   ├── demo_action_hot.npz           # Action hotkey states
│   └── demo_camera_1.npz             # Camera data index
├── run002/
│   ├── config.json
│   ├── demo_*.npz
│   └── demo_camera_1.npz
└── images/
    └── rs_rs_1_<timestamp>/
        ├── color_000000001.jpg        # run001: i.e. Image 1-50
        ├── color_000000002.jpg
        ├── ...
        ├── color_000000101.jpg        # run002: i.e. Image 101-200
        └── ...
```                                                                                                                                                                             
### 2.2. Data Postprocessing

The collected raw data needs to be converted to [Zarr](https://zarr.dev/) to store the data for training. Change the config in [`post_process_data.py`](./post_process_data.py) to match your data collection setup and execute:

```bash
python post_process_data.py
```

After postprocessing, you may see the following structure:
```yaml
# For dual camera configuration (webcam+realsense)
data/metaquest_dataset.zarr/
└── data/
├── demo_0/
│   ├── obs/
│   │   ├── agentview0_image  (N, 224, 224, 3)  # 相机0图片
│   │   ├── agentview1_image  (N, 224, 224, 3)  # 相机1图片
│   │   ├── agentview_image   (N, 224, 224, 3)  # 主视角，内容等于 agentview0_image
│   │   ├── gripper           (N,)
│   │   ├── label             (N,)
│   │   ├── robot_eef_pose    (N, 6)
│   │   ├── robot_joint       (N, 7)
│   │   └── robot_joint_vel   (N, 7)
│   ├── actions               (N, 7)
│   ├── dones                 (N,)
│   └── rewards               (N,)
├── demo_1/
│   └── ...
└── demo_2/
└── ...

# For single realsense configuration
data/metaquest_dataset.zarr/
└── data/
├── demo_0/
│   ├── obs/
│   │   ├── agentview1_image      (N, 224, 224, 3)   # 单相机realsense图片
│   │   ├── gripper               (N,)
│   │   ├── label                 (N,)
│   │   ├── robot_eef_pose        (N, 6)
│   │   ├── robot_joint           (N, 7)
│   │   └── robot_joint_vel       (N, 7)
│   ├── actions                   (N, 7)
│   ├── dones                     (N,)
│   └── rewards                   (N,)
├── demo_1/
│   └── ...
└── demo_2/
└── ...
```

The [Zarr](https://zarr.dev/) dataset format is compatible with the XARM diffusion policy training pipeline and includes all necessary metadata for normalization and sequence sampling.


## 3. Training and Eval on a Real Robot
Make sure your Ufactory XArm robot is running and accepting command from its network interface (emergency stop button within reach at all time), and your RealSense cameras plugged in to your workstation (tested with `realsense-viewer`)

### 3.1. Training
To train a Diffusion Policy, launch training [`train.py`](./train.py) using Hydra configuration management with config, serving as a training script entry

```bash
# Core function: start the training process
# Training (default: absolute action):
python train.py --config-name=train_diffusion_unet_real_image_workspace

# Relative action training:
python train.py --config-name=train_diffusion_unet_real_image_workspace +action_type=relative

# Absolute action training:
python train.py --config-name=train_diffusion_unet_real_image_workspace +action_type=absolute

# Diffusion Policy
./train_dp.sh
```


### 3.2. Configuration System

```yaml
# Main Configuration：train_diffusion_unet_real_image_workspace.yaml 
defaults:
  - task: real_lift_image_abs  # task configuration
  - _self_

# Task configuration：real_lift_image_abs.yaml  
dataset:
 _target_: diffusion_policy.dataset.real_image_dataset.RealImageDataset

# Core components
Data path: data/metaquest_xarm_dataset.zarr 
Workspace: TrainDiffusionUnetImageWorkspace
Policy: DiffusionUnetImagePolicy 
```

### 3.3. Evaluation

1. (Optional) Refer to `vcamera_server_ip` and `vcamera_server_port` in the task config file and start the corresponding vcamera server
   ```bash
   # launch camera_node
   python camera_node.py --camera-ref rs_1 --use-rgb --img-w 640 --img-h 480 --fps 30 --visualization

   # Webcam example
   python camera_node.py --camera-ref webcam_0 --use-rgb --visualization --img-h 1080 --img-w 1920 --fps 30 --publish-freq 50 --camera-address '/dev/video2'
   
### 3.4.  RealSense camera example
python camera_node.py --camera-ref rs_1 --use-rgb  --visualization --img-h 480 --img-w 640 --fps 30 --publish-freq 50

   ```
 1. Modify [eval.sh](eval.sh) to set the task and model you want to evaluate
   and run the command in separate terminals.
Assuming the training has finished and you have a checkpoint at `data/outputs/blah/checkpoints/latest.ckpt`, launch the evaluation script with:

```bash
# Core function: start the evaluation process
   # start camera node launcher
   python camera_node_launcher.py task=[task_config_file_name]
   # start inference
   python eval_real_ufact_robots.py --config-name=eval_diffusion_unet_real_image_workspace ckpt_path=path/to/your_ckpt.pth
```

Press "C" to start evaluation (handing control over to the policy). Press "S" to stop the current episode.



## 4. Codebase Tutorial
This codebase is structured under the requirement that:
1. implementing `N` tasks and `M` methods will only require `O(N+M)` amount of code instead of `O(N*M)`
2. while retaining maximum flexibility.

To achieve this requirement, we 
1. maintained a simple unified interface between tasks and methods and 
2. made the implementation of the tasks and the methods independent of each other. 

These design decisions come at the cost of code repetition between the tasks and the methods. However, we believe that the benefit of being able to add/modify task/methods without affecting the remainder and being able understand a task/method by reading the code linearly outweighs the cost of copying and pasting.

### The Split
On the task side, we have:
* `Dataset`: adapts a (third-party) dataset to the interface.
* `EnvRunner`: executes a `Policy` that accepts the interface and produce logs and metrics.
* `config/task/<task_name>.yaml`: contains all information needed to construct `Dataset` and `EnvRunner`.
* (optional) `Env`: an `gym==0.21.0` compatible class that encapsulates the task environment.

On the policy side, we have:
* `Policy`: implements inference according to the interface and part of the training process.
* `Workspace`: manages the life-cycle of training and evaluation (interleaved) of a method. 
* `config/<workspace_name>.yaml`: contains all information needed to construct `Policy` and `Workspace`.

## 5. The Interface

### 5.1. Image Policy & Dataset

A [`DiffusionUnetImagePolicy`](.diffusion_policy/policy/diffusion_unet_image_policy.py) (subclass of BaseImagePolicy) takes an observation dictionary, e.g.:

"agentview_image": Tensor of shape (B, To, C, H, W)
"robot_eef_pose": Tensor of shape (B, To, 7)
"robot_joint": Tensor of shape (B, To, N)
"robot_joint_vel": Tensor of shape (B, To, N)
"gripper": Tensor of shape (B, To, 1)
and predicts an action dictionary:

"action": Tensor of shape (B, Ta, Da)
(where Da=7 for 6D pose + gripper, Ta is action horizon)

A [`RealImageDataset`](.diffusion_policy/dataset/real_image_dataset.py) returns a sample dictionary:

"obs": Dict with keys above, each of shape (To, ...)
"action": Tensor of shape (Ta, Da)
Its get_normalizer method returns a LinearNormalizer with keys "agentview_image", "robot_eef_pose", "robot_joint", "robot_joint_vel", "gripper", and "action".

The DiffusionUnetImagePolicy handles normalization on GPU with its own copy of the LinearNormalizer. The parameters of the LinearNormalizer are saved as part of the policy's checkpoint.

### 5.2. Action Representation

The action can be either absolute (default) or relative (set use_relative_action: true in config or +action_type=relative in command line).
Relative action: All actions in a chunk are represented with respect to the last frame of the chunk (first suggested in [UMI](https://umi-gripper.github.io/))。
The dataset and policy will automatically switch between absolute and relative action modes according to the config, no manual intervention needed.






## 6. Key Components
### `Workspace`
A `Workspace` object encapsulates all states and code needed to run an experiment. 
* Inherits from [`BaseWorkspace`](./diffusion_policy/workspace/base_workspace.py).
* A single `OmegaConf` config object generated by `hydra` should contain all information needed to construct the Workspace object and running experiments. This config correspond to `config/<workspace_name>.yaml` + hydra overrides.
* The `run` method contains the entire pipeline for the experiment.
* Checkpoints happen at the `Workspace` level. All training states implemented as object attributes are automatically saved by the `save_checkpoint` method.
* All other states for the experiment should be implemented as local variables in the `run` method.

The entrypoint for training is `train.py` which uses `@hydra.main` decorator. Read [hydra](https://hydra.cc/)'s official documentation for command line arguments and config overrides. For example, the argument `task=<task_name>` will replace the `task` subtree of the config with the content of `config/task/<task_name>.yaml`, thereby selecting the task to run for this experiment.

### `Dataset`
A `Dataset` object:
* Inherits from `torch.utils.data.Dataset`.
* Returns a sample conforming to [the interface](#the-interface) depending on whether the task has Low Dim or Image observations.
* Has a method `get_normalizer` that returns a `LinearNormalizer` conforming to [the interface](#the-interface).

Normalization is a very common source of bugs during project development. It is sometimes helpful to print out the specific `scale` and `bias` vectors used for each key in the `LinearNormalizer`.

Most of our implementations of `Dataset` uses a combination of [`ReplayBuffer`](#replaybuffer) and [`SequenceSampler`](./diffusion_policy/common/sampler.py) to generate samples. Correctly handling padding at the beginning and the end of each demonstration episode according to `To` and `Ta` is important for good performance. Please read our [`SequenceSampler`](./diffusion_policy/common/sampler.py) before implementing your own sampling method.

### `Policy`
A `Policy` object:
* Inherits from `BaseLowdimPolicy` or `BaseImagePolicy`.
* Has a method `predict_action` that given observation dict, predicts actions conforming to [the interface](#the-interface).
* Has a method `set_normalizer` that takes in a `LinearNormalizer` and handles observation/action normalization internally in the policy.
* (optional) Might has a method `compute_loss` that takes in a batch and returns the loss to be optimized.
* (optional) Usually each `Policy` class correspond to a `Workspace` class due to the differences of training and evaluation process between methods.

### `EnvRunner`
A `EnvRunner` object abstracts away the subtle differences between different task environments.
* Has a method `run` that takes a `Policy` object for evaluation, and returns a dict of logs and metrics. Each value should be compatible with `wandb.log`. 

To maximize evaluation speed, we usually vectorize environments using our modification of [`gym.vector.AsyncVectorEnv`](./diffusion_policy/gym_util/async_vector_env.py) which runs each individual environment in a separate process (workaround python GIL). 

⚠️ Since subprocesses are launched using `fork` on linux, you need to be specially careful for environments that creates its OpenGL context during initialization (e.g. robosuite) which, once inherited by the child process memory space, often causes obscure bugs like segmentation fault. As a workaround, you can provide a `dummy_env_fn` that constructs an environment without initializing OpenGL.

### `ReplayBuffer`
The [`ReplayBuffer`](./diffusion_policy/common/replay_buffer.py) is a key data structure for storing a demonstration dataset both in-memory and on-disk with chunking and compression. It makes heavy use of the [`zarr`](https://zarr.readthedocs.io/en/stable/index.html) format but also has a `numpy` backend for lower access overhead.

On disk, it can be stored as a nested directory (e.g. `data/pusht_cchi_v7_replay.zarr`) or a zip file (e.g. `data/robomimic/datasets/square/mh/image_abs.hdf5.zarr.zip`).

Due to the relative small size of our datasets, it's often possible to store the entire image-based dataset in RAM with [`Jpeg2000` compression](./diffusion_policy/codecs/imagecodecs_numcodecs.py) which eliminates disk IO during training at the expense increasing of CPU workload.

Example:
```
data/pusht_cchi_v7_replay.zarr
 ├── data
 │   ├── action (25650, 2) float32
 │   ├── img (25650, 96, 96, 3) float32
 │   ├── keypoint (25650, 9, 2) float32
 │   ├── n_contacts (25650, 1) float32
 │   └── state (25650, 5) float32
 └── meta
     └── episode_ends (206,) int64
```

Each array in `data` stores one data field from all episodes concatenated along the first dimension (time). The `meta/episode_ends` array stores the end index for each episode along the fist dimension.

### `SharedMemoryRingBuffer`
The [`SharedMemoryRingBuffer`](./diffusion_policy/shared_memory/shared_memory_ring_buffer.py) is a lock-free FILO data structure used extensively in our [real robot implementation](./diffusion_policy/real_world) to utilize multiple CPU cores while avoiding pickle serialization and locking overhead for `multiprocessing.Queue`. 

As an example, we would like to get the most recent `To` frames from 5 RealSense cameras. We launch 1 realsense SDK/pipeline per process using [`SingleRealsense`](./diffusion_policy/real_world/single_realsense.py), each continuously writes the captured images into a `SharedMemoryRingBuffer` shared with the main process. We can very quickly get the last `To` frames in the main process due to the FILO nature of `SharedMemoryRingBuffer`.

We also implemented [`SharedMemoryQueue`](./diffusion_policy/shared_memory/shared_memory_queue.py) for FIFO, which is used in [`RTDEInterpolationController`](./diffusion_policy/real_world/rtde_interpolation_controller.py).

### `RealEnv`
In contrast to [OpenAI Gym](https://gymnasium.farama.org/), our polices interact with the environment asynchronously. In [`RealEnv`](./diffusion_policy/real_world/real_env.py), the `step` method in `gym` is split into two methods: `get_obs` and `exec_actions`. 

The `get_obs` method returns the latest observation from `SharedMemoryRingBuffer` as well as their corresponding timestamps. This method can be call at any time during an evaluation episode.

The `exec_actions` method accepts a sequence of actions and timestamps for the expected time of execution for each step. Once called, the actions are simply enqueued to the `RTDEInterpolationController`, and the method returns without blocking for execution.

## 🩹 Adding a Task
Read and imitate:
* `diffusion_policy/dataset/pusht_image_dataset.py`
* `diffusion_policy/env_runner/pusht_image_runner.py`
* `diffusion_policy/config/task/pusht_image.yaml`

Make sure that `shape_meta` correspond to input and output shapes for your task. Make sure `env_runner._target_` and `dataset._target_` point to the new classes you have added. When training, add `task=<your_task_name>` to `train.py`'s arguments.

## 🩹 Adding a Method
Read and imitate:
* `diffusion_policy/workspace/train_diffusion_unet_image_workspace.py`
* `diffusion_policy/policy/diffusion_unet_image_policy.py`
* `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml`




# Acknowledgement
Our work is built upon
[Diffusion Policy](https://github.com/real-stanford/diffusion_policy),
[Stable Diffusion](https://github.com/CompVis/stable-diffusion),
[UMI](https://github.com/real-stanford/universal_manipulation_interface)
Thanks for their great work!




### debug
