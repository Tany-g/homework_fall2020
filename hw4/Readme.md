# 1.安装环境

## 1.1安装rlgpu环境
先安装rlgpu环境，官方链接 https://developer.nvidia.com/isaac-gym

官方给的torch版本为1.8 在运行某些task的时候回报错 我改成了1.13
```bash
conda create -n rlgpu python=3.7

conda activate rlgpu
  - python=3.7
  - pytorch=1.13.1+cu117
  - torchvision=0.14.1+cu117
  - pyyaml>=5.3.1
  - scipy>=1.5.0
  - tensorboard=2.11.2
cd python
pip install -e .
```
保证环境能 运行
```bash
python python/examples/joint_monkey.py
```
## 1.2安装isaacgymenv
clone GIT仓库IsaacGymEnvs到本地
在原来rlgpu的环境下
```bash
cd IsaacGymEnvs
```
安装环境IsaacGymEnv
```bash
pip install -e .
```
## 1.3安装cs285
在当前环境下安装cs285

```bash
cd homework_fall2020/hw4
pip install -e .
```
# 2.运行code

```bash
python cs285/scripts/run_hw4_mb.py --exp_name q1_cheetah_n500_arch1x32 --env_name cheetah-cs285-v0 --add_sl_noise --n_iter 3 --batch_size_initial 20000 --num_agent_train_steps_per_iter 500 --n_layers 1 --size 32 --scalar_log_freq -1 --video_log_freq -1
```


