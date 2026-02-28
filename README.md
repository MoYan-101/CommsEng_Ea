# Catalyst ML Pipeline
This repo trains ML models to predict catalyst performance using numeric features
plus material and text embeddings. It includes training, inference, and
visualization in one pipeline.

## Main Workflow
1. Data loading and preprocessing (`data_preprocessing/data_loader_modified.py`)
   - Cleans CSV, normalizes missing tokens, builds numeric + material/text embeddings.
   - Generates feature matrix `X`, targets `Y`, and metadata.
2. Training (`train.py`)
   - Optional Optuna tuning.
   - 5-fold CV metrics and model training for RF/DT/CatBoost/XGB/ANN/SVM.
   - Saves models, scalers, and SHAP data.
3. Inference (`inference.py`)
   - Loads trained models and metadata.
   - Generates 2D heatmaps, optional 3D heatmaps, and confusion-like matrices.
4. Visualization (`visualization.py`)
   - Plots CV metrics, residuals, SHAP, heatmaps, and other figures.

## Quick Start
1) Edit `configs/config.yaml` (data path, model list, Optuna settings).
2) Run:
```bash
bash run.sh
```

## Run Options
`run.sh` will prompt for `overfit_penalty_alpha` values.
- Enter one or multiple values (comma-separated), e.g. `0.0,0.03`.
- Press Enter to use the value in `configs/config.yaml`.
- Each alpha gets its own `RUN_ID` suffix to avoid overwriting results.

You can skip the prompt:
```bash
OVERFIT_ALPHA_LIST="0.0,0.03" bash run.sh
```

## Max Performance (CPU)
`run.sh` now auto-detects CPU threads and applies a high-throughput runtime profile:
- parallel Optuna trials: `OPTUNA_N_JOBS` (auto heuristic: 16 threads CPU -> 4 trials)
- model-level threads: `MODEL_N_JOBS`, `CATBOOST_THREAD_COUNT`, `XGB_N_JOBS`, `SVM_N_JOBS`
- torch threads: `TORCH_NUM_THREADS`, `TORCH_NUM_INTEROP_THREADS`
- BLAS/OpenMP anti-oversubscription defaults (`OMP_NUM_THREADS=1`, etc.)

For full manual control:
```bash
CPU_TOTAL=16 \
OPTUNA_N_JOBS=4 \
MODEL_N_JOBS=16 \
CATBOOST_THREAD_COUNT=16 \
XGB_N_JOBS=16 \
SVM_N_JOBS=16 \
TORCH_NUM_THREADS=16 \
TORCH_NUM_INTEROP_THREADS=4 \
OVERFIT_ALPHA_LIST="0.0" \
bash run.sh
```

## Config Highlights
- `data.path`: CSV file path.
- `data_loader.impute_method`: `kde` or `simple` missing value handling.
- `data_loader.preserve_null`: keep "Null" as a valid category.
- `data_loader.element_embedding`: `advanced` (uses AdvancedMaterialFeaturizer) or `simplified`/`basic`.
- `data_loader.promoter_ratio_cols`: ratio columns aligned with `element_cols` (Promoter 1/2).
- `data_loader.promoter_onehot`: add promoter identity one-hot features.
- `preprocessing.standardize_all_features`: scale all feature columns (numeric + embeddings + one-hot).
- `optuna.overfit_penalty_alpha`: overfit penalty weight.
- `inference.heatmap_axes`: 2D or 3D axes for heatmaps.
- `inference.enable_3d_heatmap`: toggle 3D heatmap generation.
- `inference.skip_3d_models`: skip 3D heatmap for listed models (e.g. SVM).

## Data Processing Details (current pipeline)
The default pipeline is implemented in `data_preprocessing/data_loader_modified.py`:

1) **Input columns**
   - Uses all columns except Y and metadata (`DOI`, `Name`, `Year` by default).
   - Y columns default to:
     - `CO selectivity (%)`
     - `Methanol selectivity (%)`
     - `STY_CH3OH (g/kg·h) (LN scale)`
     - `CO2 conversion efficiency (%)`

2) **Missing handling**
   - Common missing tokens (e.g., `NaN`, `None`, empty) → `np.nan`.
   - `"Null"` is **not** treated as missing; it is a valid category.
   - Optional KDE or simple imputation for missing values (`data_loader.impute_method`).

3) **Promoter features (per Promoter column)**
   - **Base material vector** via `AdvancedMaterialFeaturizer` (requires `pymatgen` + `matminer`).
   - **Ratio-weighted vector**: `magpie * promoter_ratio`.
   - **Explicit Null flag**: `is_null` (1 if value is `"Null"`).
   - **Optional one-hot identity** (`promoter_onehot: true`).
   - Final promoter block: `[magpie | magpie*ratio | is_null | onehot]`.

4) **Text features**
   - Text columns are encoded as **one-hot** (preserves discrete synthesis types).

5) **Scaling**
   - With `preprocessing.standardize_all_features: true`, all feature columns
     (numeric + embeddings + one-hot) are standardized together.

## Outputs
- `models/<csv_name>/<run_id>/...` trained models, scalers, metadata.
- `postprocessing/<csv_name>/<run_id>/...` training artifacts and inference arrays.
- `evaluation/figures/<csv_name>/<run_id>/...` plots and summaries.

## Notes
- 3D heatmaps can be slow. Use `skip_3d_models` and `max_combinations` to control cost.
- SVM SHAP uses KernelExplainer and can be slow on large datasets.

## 对数变换与展示尺度说明（当前配置）
以下逻辑对应当前 `configs/config.yaml`。

1. 基础规则
- 仅对 `data_loader.log_transform_cols` 指定的数值列做对数变换，公式为：
  `x -> ln(max(x, eps))`，其中 `eps = data_loader.log_transform_eps`（当前为 `1e-8`）。
- 列名中已明确标注 `(LN scale)` 的数据，视为“已在对数域”，不再重复做 `ln`。

2. 当前会做对数变换的列
- 全模型（`RF/DT/CatBoost/XGB/ANN/SVM`）：
  - `Calcination time (h)`
  - `Molar ratio (Zn:Cu)`
- 仅额外对 `ANN/SVM` 生效（`log_transform_cols_extra_for`）：
  - `H2/CO2 ratio (-)`
  - `Promoter 1 ratio (Promoter 1:Cu)`
  - `Promoter 2 ratio (Promoter 2:Cu)`

3. 当前已是 LN scale、不会再做对数变换的列
- `Catalyst surface area (m2/g) (LN scale)`
- `GHSV (mL/g.h) (LN scale)`

4. 训练阶段
- 在数据加载阶段先完成上述 `ln(max(x, eps))` 处理。
- 随后再进行训练/验证划分与标准化。
- 模型学习发生在“对数后 + 标准化后”的输入空间中。

5. 推断与热图阶段
- 外部可视化网格（`grid_x/grid_y/grid_z`）按原始物理尺度生成与保存。
- 若某轴属于对数列，则仅在送入模型前临时执行同样的 `ln(max(x, eps))`。
- 然后再进入与训练一致的标准化与预测流程，保证输入域一致且外部显示保持原始单位。

6. SHAP 展示阶段
- SHAP 值本身仍对应模型输入域。
- 用于展示的特征值会先逆标准化，再对对数列执行 `exp` 还原到原始尺度。
- 同时保留模型域副本用于追踪（例如 `X_full_model_domain`），并额外保存展示域数据文件。
