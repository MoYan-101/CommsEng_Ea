# SLURM 使用说明

这个项目的完整入口流程是：

- `run.sh` -> `train.py` -> `inference.py` -> `visualization.py`

可用的 SLURM 模板：

- `slurm/run_pipeline_cpu.sbatch`
- `slurm/run_pipeline_array_cpu.sbatch`（通过 job array 并行多个 alpha）

## 1）集群上一次性环境准备

在项目根目录执行：

```bash
cd /path/to/case_alpha_0_0_colab
python -m pip install -r requirements.txt
```

如果集群使用 conda，建议创建/使用 `comms310`（与 `run.sh` 默认一致）。

## 2）检查数据路径

当前数据文件路径配置在：

- `configs/config.yaml`

请确保 `data.path` 在计算节点可访问。

## 3）提交单个完整流水线任务

```bash
cd /path/to/case_alpha_0_0_colab
sbatch --export=ALL,CONDA_ENV=comms310,OVERFIT_ALPHA_LIST=0.0 slurm/run_pipeline_cpu.sbatch
```

默认单任务资源：`16 CPU / 64G / 120h`。

也可以在一个任务里串行跑多个 alpha：

```bash
sbatch --export=ALL,CONDA_ENV=comms310,OVERFIT_ALPHA_LIST=0.0,0.03,0.05 slurm/run_pipeline_cpu.sbatch
```

## 3.1）提交一个 job array（并行多个 alpha）

脚本内置默认 alpha：

- `0.0,0.01,0.03,0.05,0.07`

一次提交 5 个 alpha（默认最多同时运行 4 个，见 `--array=0-4%4`）：

```bash
cd /path/to/case_alpha_0_0_colab
sbatch --export=ALL,CONDA_ENV=comms310 slurm/run_pipeline_array_cpu.sbatch
```

默认每个 array 子任务资源：`8 CPU / 32G / 120h`。

自定义 alpha 示例：

```bash
export ALPHA_LIST="0.0,0.02,0.04"
sbatch --array=0-2%3 --export=ALL,CONDA_ENV=comms310 slurm/run_pipeline_array_cpu.sbatch
```

## 4）查看任务状态和日志

```bash
squeue -u "$USER"
tail -f slurm-MY-CE1-<jobid>.out
# array 任务日志文件名示例：
tail -f slurm-MY-CE1-<jobid>_<taskid>.out
```

## 5）注意事项

- 如果不设置 `OVERFIT_ALPHA_LIST`，`run.sh` 会进入交互输入模式；在 SLURM 中务必设置。
- CPU 资源由 `--cpus-per-task` 决定，并会映射到 `MODEL_N_JOBS`、`XGB_N_JOBS` 等并行参数。
- 如果你要在同一目录并发提交多个普通任务（非 array），建议显式给唯一 `RUN_ID`：

```bash
sbatch --export=ALL,CONDA_ENV=comms310,OVERFIT_ALPHA_LIST=0.0,RUN_ID=slurm_$(date +%Y%m%d_%H%M%S) slurm/run_pipeline_cpu.sbatch
```

- 在 array 脚本中，已自动关闭并发保护（已设置 `ALLOW_CONCURRENT_PIPELINE=1` 和 `ALLOW_CONCURRENT_TRAIN=1`）。
