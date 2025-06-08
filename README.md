<img src="docs/media/Overview.png" alt="fig1" />

---

# Demonstrating an End-to-End Sim-to-Real Transfer for Lidar-Based Navigation Using Reinforcement Learning

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.0.2-silver.svg)](https://isaac-sim.github.io/IsaacLab/v2.0.0/)
[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/20.04/)

Environments, assets, workflow for open-source mobile robotics, integrated with IsaacLab.

# Manual for installing the Environment

The following subsections explain how to install the environment to run the code. 

## Installing IsaacLab (~20 min)

Note: Only use this pip installation approach if you're on Ubuntu 22.04+ or Windows. For Ubuntu 20.04, install from the binaries. [link](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

WheeledLab is built atop Isaac Lab. If you do not yet have Isaac Lab installed, it is open-source and installation instructions for Isaac Sim v4.5.0 and Isaac Lab v2.0.2 can be found below:

```bash
# Create a conda environment named WL and install Isaac Sim v4.5.0 in it:
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # Or `pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118` for CUDA 11
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Install Isaac Lab v2.0.2 (make sure you have build dependencies first, e.g. `sudo apt install cmake build-essential` on ubuntu)
git clone --branch v2.0.2 https://github.com/isaac-sim/IsaacLab.git
./isaaclab.sh -i
```

Source: https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html

If you already have IsaacLab you can skip this and instead head [here](#create-new-isaaclab-conda-environment) to set up a new conda environment for this repository.

### Create New IsaacLab Conda Environment

We recommend setting up a new [conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) environment to include both `IsaacLab` packages and `WheeledLab` packages. You can do this using Isaac Lab's convenient setup scripts:

```bash
cd <IsaacLab>
./isaaclab.sh --conda env_isaaclab
conda activate env_isaaclab
./isaaclab.sh -i
```

## Installing WheeledLab (~5 min)

```bash
# Activate the conda environment that was created via the IsaacLab setup.
conda activate <your IsaacLab env here> # 'env_isaaclab' if you followed instructions above

git clone git@github.com:kgulikers/Team_internship_5ARIP10.git
```

After this, we recommend [Setting Up VSCode](https://github.com/UWRobotLearning/WheeledLab?tab=readme-ov-file#training-quick-start).

## Training Quick Start

Training runs can take a couple hours to produce a transferable policy.

To start a navigatio run us the following command:

```bash
python source/wheeledlab_rl/scripts/train_rl.py --headless -r RSS_NAV_CONFIG
```

Though optional (and free), we strongly advise using [Weights & Biases](https://wandb.ai/site/) (`wandb`) to record and track training status. Logging to `wandb` is turned on by default. If you would like to disable it, add `train.log.no_wandb=True` to the CLI arguments.



## References

### This work

```
@misc{2502.07380,
Author = {Tyler Han and Preet Shah and Sidharth Rajagopal and Yanda Bao and Sanghun Jung and Sidharth Talia and Gabriel Guo and Bryan Xu and Bhaumik Mehta and Emma Romig and Rosario Scalise and Byron Boots},
Title = {Demonstrating WheeledLab: Modern Sim2Real for Low-cost, Open-source Wheeled Robotics},
Year = {2025},
Eprint = {arXiv:2502.07380},
}
```

### Cited

[1] Sidharth Talia, Matt Schmittle, Alexander Lambert, Alexander Spitzer, Christoforos Mavrogiannis, and Siddhartha S. Srinivasa.Demonstrating HOUND: A Low-cost Research Platform for High-speed Off-road Underactuated Nonholonomic Driving, July 2024.URL http://arxiv.org/abs/2311.11199.arXiv:2311.11199 [cs].

[2] Siddhartha S. Srinivasa, Patrick Lancaster, Johan Michalove, Matt Schmittle, Colin Summers, Matthew Rockett, Rosario Scalise, Joshua R. Smith, Sanjiban Choudhury, Christoforos Mavrogiannis, and Fereshteh Sadeghi.MuSHR: A Low-Cost, Open-Source Robotic Racecar for Education and Research, December 2023.URL http://arxiv.org/abs/1908.08031.arXiv:1908.08031 [cs].

[3] Matthew O’Kelly, Hongrui Zheng, Dhruv Karthik, and Rahul Mangharam. F1TENTH: An Open-source Eval- uation Environment for Continuous Control and Reinforcement Learning. In Proceedings of the NeurIPS 2019 Competition and Demonstration Track, pages 77– 89. PMLR, August 2020. URL https://proceedings.mlr. press/v123/o-kelly20a.html. ISSN: 2640-3498.
