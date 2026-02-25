# SLURM 使用说明（CommsEng）

本文档是当前项目的标准提交流程（现在的主目录是 `CommsEng`）。

完整流水线入口：

- `run.sh` -> `train.py` -> `inference.py` -> `visualization.py`

可用的 SLURM 脚本：

- `slurm/run_pipeline_cpu.sbatch`（单任务）
- `slurm/run_pipeline_array_cpu.sbatch`（job array 并行多个 alpha）

## 1）登录后第一步（每次会话）

```bash
cd /path/to/CommsEng
pwd
chmod +x run.sh slurm/*.sbatch
mkdir -p logs
```

## 2）环境准备（首次或环境丢失时）

推荐 conda 环境名：`mlcpu`。

```bash
cd /path/to/CommsEng
conda create -n mlcpu python=3.10 -y
conda activate mlcpu
python -m pip install -U pip
python -m pip install -r requirements.txt
```

说明：`sbatch` 脚本里已经包含了非交互 shell 的环境激活逻辑，默认优先 `micromamba`，再回退 `conda`（会尝试 `conda.sh`），最后回退 `.venv`。
默认 `CONDA_ENV=mlcpu`。如果未来你换环境名，提交时用 `--export=ALL,CONDA_ENV=<你的环境名>` 覆盖即可。
`sbatch` 已同时支持 `micromamba` 和 `conda` 激活（优先 `micromamba`，其次 `conda`）。

如果你已经创建过 `mlcpu` 环境，则无需重复创建，直接激活并验证即可：

```bash
conda activate mlcpu
python -V
python -c "import sys; print(sys.executable)"
```

## 2.1）安装前检查 Python 版本

在安装依赖前，先确认当前 `python` 不是系统 Python 3.6：

```bash
which python
python -V
python -c "import sys; print(sys.executable)"
python -m pip -V
```

期望结果：

- Python 版本是 `3.10.x`（或至少 `>=3.9`）。
- `sys.executable` 指向你的 conda 环境（如 `.../envs/mlcpu/bin/python`）。
- 不应是 `/usr/bin/python3` 或 Python 3.6。

如果不是期望结果，先执行：

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh
conda activate mlcpu
```

## 3）提交前 30 秒自检

```bash
cd /path/to/CommsEng
grep -n "path:" configs/config.yaml
ls -lh data/Main_20260128_cleansed.csv
```

请确保 `configs/config.yaml` 的 `data.path` 在计算节点可访问。

## 4）提交单任务（CPU）

默认资源（已写入脚本）：`16 CPU / 64G / 120h`。
当前脚本默认环境就是 `mlcpu`，所以 `CONDA_ENV` 可省略；显式写上更清晰。

```bash
cd /path/to/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

一个任务串行跑多个 alpha：

```bash
cd /path/to/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,0.03,0.05 slurm/run_pipeline_cpu.sbatch
```

## 5）提交 array（推荐并行模式）

默认 alpha 列表：`0.0,0.01,0.03,0.05,0.07`

默认资源（每个子任务）：`8 CPU / 32G / 120h`  
默认并发上限：`--array=0-4%4`（5 个任务最多同时跑 4 个）
当前脚本默认环境就是 `mlcpu`，所以 `CONDA_ENV` 可省略；显式写上更清晰。

```bash
cd /path/to/CommsEng
mkdir -p logs
sbatch --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

自定义 alpha：

```bash
cd /path/to/CommsEng
export ALPHA_LIST="0.0,0.02,0.04"
mkdir -p logs
sbatch --array=0-2%3 --export=ALL,CONDA_ENV=mlcpu slurm/run_pipeline_array_cpu.sbatch
```

## 6）查看运行状态与日志

```bash
squeue -u "$USER"
squeue -u "$USER" -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"
```

看日志：

```bash
# 单任务
tail -f logs/slurm-MY-CE1-<jobid>.out

# array 子任务
tail -f logs/slurm-MY-CE1-<jobid>_<taskid>.out
```

## 7）结果目录

每个 alpha/run_id 的产物主要在：

- `models/<csv_name>/<run_id>/...`
- `postprocessing/<csv_name>/<run_id>/...`
- `evaluation/figures/<csv_name>/<run_id>/...`

## 8）常见问题

- `OVERFIT_ALPHA_LIST` 没设置：`run.sh` 会进入交互输入，SLURM 里务必通过 `--export` 或 array 传入。
- 想并发多个普通任务（非 array）：建议显式设置唯一 `RUN_ID`，避免结果覆盖：

```bash
sbatch --export=ALL,CONDA_ENV=mlcpu,OVERFIT_ALPHA_LIST=0.0,RUN_ID=slurm_$(date +%Y%m%d_%H%M%S) slurm/run_pipeline_cpu.sbatch
```

- array 脚本已自动关闭并发保护（已设置 `ALLOW_CONCURRENT_PIPELINE=1` 与 `ALLOW_CONCURRENT_TRAIN=1`）。
- `minepy` 编译失败（常见于老系统/缺少编译工具链）：可先跳过 `minepy`，其功能会自动回退到 `dcor/Pearson`。更新代码后直接执行：

```bash
python -m pip install --prefer-binary -r requirements.txt
```
