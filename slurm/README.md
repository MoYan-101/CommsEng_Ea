# SLURM 使用说明（CommsEng）

本文档按你当前集群状态整理，可直接照抄执行。

流水线入口：

- `run.sh` -> `train.py` -> `inference.py` -> `visualization.py`

可用脚本：

- `slurm/run_pipeline_cpu.sbatch`（单任务）
- `slurm/run_pipeline_array_cpu.sbatch`（array 并行多个 alpha）

## 0）当前集群基线（已对齐）

- `normal` 分区：每节点 `28 CPU`、约 `128660 MB` 内存。
- `MaxArraySize=1001`，调度器 `sched/backfill`。
- 当前脚本默认：
- 单任务：`16 CPU / 64G / 120h`
- array 子任务：`8 CPU / 32G / 120h`
- array 并发：`--array=0-4%5`（默认 5 个 alpha 全并发）

## 1）登录后第一步（每次会话）

```bash
cd /public/home/user2/CommsEng
pwd
chmod +x run.sh slurm/*.sbatch
mkdir -p logs
```

说明：`logs/` 只需建一次；重复执行 `mkdir -p logs` 不会有副作用。

## 2）环境准备（首次或环境丢失时）

推荐环境名：`mlcpu`，推荐优先用 `micromamba`。

```bash
cd /public/home/user2/CommsEng
eval "$($HOME/bin/micromamba shell hook --shell=bash)"
micromamba create -n mlcpu python=3.10 -y
micromamba activate mlcpu
python -m pip install -U pip setuptools wheel
python -m pip install --prefer-binary -r requirements.txt
```

脚本内环境激活逻辑：

- 先找 `micromamba`（`PATH`、`~/bin/micromamba`、`~/.local/bin/micromamba`）。
- 再回退 `conda`（自动尝试 `conda.sh`）。
- 最后回退项目内 `.venv`。

默认 `CONDA_ENV=mlcpu`。如需换环境名，提交时覆盖：

```bash
sbatch --export=ALL,CONDA_ENV=<你的环境名> ...
```

如果你的环境是按绝对路径管理（例如 `/public/home/user2/miniconda3/envs/mlcpu`），也可以直接传路径：

```bash
sbatch --export=ALL,CONDA_ENV=/public/home/user2/miniconda3/envs/mlcpu ...
```

## 2.1）安装前检查 Python 版本（防坑）

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
python -m pip -V
```

期望：

- Python 是 `3.10.x`（或至少 `>=3.9`）。
- `sys.executable` 指向 `.../envs/mlcpu/bin/python`。
- 不是 `/usr/bin/python3`（系统 Python）。

## 3）提交前 30 秒自检

```bash
cd /public/home/user2/CommsEng
grep -n "path:" configs/config.full.yaml
# 或测试配置
grep -n "path:" configs/config.test.yaml
ls -lh data/Main_20260128_cleansed.csv
```

## 4）提交单任务（CPU）

```bash
cd /public/home/user2/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

选择配置（默认 `full`）：

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,CONFIG_PROFILE=test,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

或传入自定义配置文件：

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,CONFIG_PATH=configs/config.test.yaml,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

单任务串行多个 alpha：

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,0.03,0.05 slurm/run_pipeline_cpu.sbatch
```

## 5）提交 array（推荐）

默认 alpha 列表：`0.0,0.01,0.03,0.05,0.07`

```bash
cd /public/home/user2/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

自定义 alpha：

```bash
export ALPHA_LIST="0.0,0.02,0.04"
sbatch --array=0-2%3 --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

## 6）查看状态与日志

```bash
squeue -u "$USER"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

```bash
tail -f logs/slurm-MY-CE-Ea-<jobid>.out
tail -f logs/slurm-MY-CE-Ea-<jobid>_<taskid>.out
```

## 7）结果目录

- `models/<csv_name>/<run_id>/...`
- `postprocessing/<csv_name>/<run_id>/...`
- `evaluation/figures/<csv_name>/<run_id>/...`

## 8）常见问题

- 未传 `OVERFIT_ALPHA_LIST`：`run.sh` 会进入交互输入，SLURM 下应通过 `--export` 或 array 传入。
- 多个普通任务并发时，建议给唯一 `RUN_ID`，避免覆盖：

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,RUN_ID=slurm_$(date +%Y%m%d_%H%M%S) slurm/run_pipeline_cpu.sbatch
```

- array 脚本已设置 `ALLOW_CONCURRENT_PIPELINE=1` 和 `ALLOW_CONCURRENT_TRAIN=1`。
- `minepy` 在老系统可能编译失败，可先继续跑其余流程，后续再补装。

## 9）若还要继续“精调资源”，还缺两项数据

- 一次真实作业的资源画像：`sacct -j <jobid> --format=JobID,Elapsed,TotalCPU,MaxRSS,AllocCPUS,State`
- 当前账号是否必须指定 `--account` / `--qos`（若管理员有硬性要求）

## 20260225运行：
cd /public/home/user2/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=/public/home/user2/micromamba/envs/mlcpu slurm/run_pipeline_array_cpu.sbatch

这会使用脚本默认：
--array=0-4%5
ALPHA_LIST=0.0,0.01,0.03,0.05,0.07
job name: MY-CE-Ea
