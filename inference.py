#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
inference.py

- 读取 ./models/<model_type>/trained_model.pkl / best_ann.pt
- 读取 metadata.pkl (continuous_cols, onehot_groups, group_value_vectors, feature_means …)
- 使用训练集均值作为 baseline,按组替换嵌入向量生成 confusion-like 预测
- 输出 heatmap_pred.npy, confusion_pred.npy 等
  (可加权: sum_real += real_pred * freq; avg_real = sum_real / sum_freq)
"""

import yaml
import os
import re
import traceback
import numpy as np
import torch
import joblib
from tqdm import trange
from itertools import product   # 你原来的 import 保留
import json                     # 你原来的 import 保留

from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output
from utils import get_model_dir, get_root_model_dir, get_postprocess_dir, get_run_id

# 各种模型
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

from itertools import combinations


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _write_inference_error(outdir, model_type, err):
    try:
        ensure_dir(outdir)
        err_path = os.path.join(outdir, "error.log")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(f"[ERROR] Inference failed for {model_type}:\n")
            f.write(str(err) + "\n\n")
            f.write(traceback.format_exc())
    except Exception:
        # If logging fails, fall back to console only.
        pass


def _find_latest_run_id(csv_name: str) -> str | None:
    base_dir = os.path.join("models", csv_name)
    if not os.path.isdir(base_dir):
        return None
    pattern = re.compile(r"^\d{8}_\d{6}(?:_.*)?$")
    candidates = []
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if not os.path.isdir(path):
            continue
        if not pattern.match(name):
            continue
        if not os.path.exists(os.path.join(path, "metadata.pkl")):
            continue
        candidates.append(name)
    return max(candidates) if candidates else None



# 放到 inference_main() 顶部、for-mtype 循环里 —— 在拿到 x_col_names 之后
def find_group_idx(keyword, groups, colnames):
    kw = keyword.lower()
    for idx, grp in enumerate(groups):
        if any(kw in colnames[c].lower() for c in grp):
            return idx
    return None


def find_group_idx_by_name(keyword, group_names):
    kw = keyword.lower()
    for idx, name in enumerate(group_names):
        if kw in str(name).lower():
            return idx
    return None

def _get_model_input_dim(model):
    if hasattr(model, "n_features_in_"):
        val = int(model.n_features_in_)
        return val if val > 0 else None
    if hasattr(model, "model"):
        if hasattr(model.model, "n_features_in_"):
            val = int(model.model.n_features_in_)
            return val if val > 0 else None
        if hasattr(model.model, "feature_count_"):
            val = int(model.model.feature_count_)
            return val if val > 0 else None
        if hasattr(model.model, "feature_names_"):
            names = getattr(model.model, "feature_names_", None)
            if names:
                return len(names)
    if hasattr(model, "feature_names_"):
        names = getattr(model, "feature_names_", None)
        if names:
            return len(names)
    if hasattr(model, "net"):
        for layer in model.net:
            if hasattr(layer, "in_features"):
                return int(layer.in_features)
    return None


def _assert_feature_dim(model, expected_dim, model_type):
    actual_dim = _get_model_input_dim(model)
    if actual_dim is not None and actual_dim != expected_dim:
        raise RuntimeError(
            f"[ERROR] Feature dimension mismatch for {model_type}: "
            f"model expects {actual_dim}, current input has {expected_dim}. "
            "Please retrain the model."
        )


# --------------------------------------------------
#              按类型加载已训练模型
# --------------------------------------------------
def load_inference_model(model_type, config, run_id=None):
    csv_name = os.path.splitext(os.path.basename(config["data"]["path"]))[0]
    rid = get_run_id(config) if run_id is None else run_id
    model_dir = get_model_dir(csv_name, model_type, run_id=rid)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    x_col_path = os.path.join(model_dir, "x_col_names.npy")
    y_col_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")

    x_col_names = list(np.load(x_col_path, allow_pickle=True))
    y_col_names = list(np.load(y_col_path, allow_pickle=True))

    # ---------- ANN ----------
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"].copy()

        # 若 hidden_dims 不在 ann_cfg，就尝试从 Optuna 最优参数里补
        if "hidden_dims" not in ann_cfg:
            best_params = None
            if config.get("optuna", {}).get("enable", False):
                optuna_dir = get_postprocess_dir(csv_name, rid, "optuna", "ANN")
                best_params_path = os.path.join(optuna_dir, "best_params.pkl")
                if os.path.exists(best_params_path):
                    best_params = joblib.load(best_params_path)
                    if isinstance(best_params.get("hidden_dims"), str):
                        best_params["hidden_dims"] = tuple(int(x) for x in best_params["hidden_dims"].split(","))
                    ann_cfg.update(best_params)
                    print(f"[INFO] Updated ann_params from optuna: {ann_cfg}")
                else:
                    print(f"[WARN] best_params not found for ANN => {best_params_path}, using defaults.")

            # fallback defaults for inference
            ann_cfg.setdefault("hidden_dims", (64, 64))
            ann_cfg.setdefault("dropout", 0.0)
            ann_cfg.setdefault("activation", "ReLU")
            ann_cfg.setdefault("random_seed", 42)

        net = ANNRegression(
            input_dim=len(x_col_names),
            output_dim=len(y_col_names),
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", 42)
        )

        ckpt_path = os.path.join(model_dir, "best_ann.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容旧版 torch
            state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        _assert_feature_dim(net, len(x_col_names), model_type)
        return net, x_col_names, y_col_names

    # ---------- 其余模型 ----------
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")
        model = joblib.load(pkl_path)
        _assert_feature_dim(model, len(x_col_names), model_type)
        return model, x_col_names, y_col_names


def model_predict(model, X_2d):
    """统一预测接口:Torch / Sklearn / Booster"""
    if hasattr(model, "eval") and hasattr(model, "forward"):
        with torch.no_grad():
            out = model(torch.tensor(X_2d, dtype=torch.float32)).cpu().numpy()
    else:
        out = model.predict(X_2d)

    # --- 新增：保证 2D ---
    if out.ndim == 1:
        out = out.reshape(-1, 1)

    return out


def _get_base_vector(stats_dict, x_col_names):
    base = stats_dict.get("feature_means", None)
    if base is not None:
        base = np.asarray(base, dtype=float)
        if base.shape[0] == len(x_col_names):
            return base.copy()

    base_vec = np.zeros(len(x_col_names), dtype=float)
    for cname, cstat in stats_dict.get("continuous_cols", {}).items():
        if cname in x_col_names:
            base_vec[x_col_names.index(cname)] = float(cstat.get("mean", 0.0))
    return base_vec


def _get_group_entries(stats_dict, x_col_names):
    groups = []
    onehot_groups = stats_dict.get("onehot_groups", [])
    group_names = stats_dict.get("group_names", [])
    group_value_vectors = stats_dict.get("group_value_vectors", {})

    for gid, grp in enumerate(onehot_groups):
        name = group_names[gid] if gid < len(group_names) else f"group_{gid}"
        info = group_value_vectors.get(name)
        if not info:
            continue
        vecs = np.asarray(info.get("vectors", []), dtype=float)
        if vecs.size == 0:
            continue
        if vecs.shape[1] != len(grp):
            continue
        weights = info.get("weights")
        if weights is not None and len(weights) == vecs.shape[0]:
            weights = np.asarray(weights, dtype=float)
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                weights = None
        else:
            weights = None
        values = info.get("values", list(range(vecs.shape[0])))
        groups.append(
            {
                "gid": gid,
                "name": name,
                "indices": np.asarray(grp, dtype=int),
                "vectors": vecs,
                "weights": weights,
                "values": values,
            }
        )
    return groups


def _build_combo_templates(base_vec, groups, fixed=None, max_combos=None, seed=42):
    fixed = fixed or {}
    base = base_vec.copy()
    for g in groups:
        if g["gid"] in fixed:
            base[g["indices"]] = fixed[g["gid"]]

    iter_groups = [g for g in groups if g["gid"] not in fixed]
    if not iter_groups:
        return base.reshape(1, -1), np.array([1.0], dtype=float)

    sizes = [len(g["vectors"]) for g in iter_groups]
    total = 1
    for s in sizes:
        total *= s

    rng = np.random.default_rng(seed)
    if max_combos and total > max_combos:
        n = int(max_combos)
        templates = np.repeat(base.reshape(1, -1), n, axis=0)
        for g in iter_groups:
            w = g["weights"]
            if w is None:
                idxs = rng.integers(0, len(g["vectors"]), size=n)
            else:
                idxs = rng.choice(len(g["vectors"]), size=n, p=w)
            templates[:, g["indices"]] = g["vectors"][idxs]
        weights = np.ones(n, dtype=float)
        return templates, weights

    from itertools import product

    templates = []
    weights = []
    for combo in product(*[range(len(g["vectors"])) for g in iter_groups]):
        vec = base.copy()
        w = 1.0
        for g, idx in zip(iter_groups, combo):
            vec[g["indices"]] = g["vectors"][idx]
            if g["weights"] is not None:
                w *= float(g["weights"][idx])
        templates.append(vec)
        weights.append(w)

    templates = np.vstack(templates)
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        weights = np.ones_like(weights)
    return templates, weights


def _weighted_predict(model, batch, weights, scaler_x, scaler_y, scale_cols_idx):
    X = batch.copy()
    if scaler_x is not None:
        X[:, scale_cols_idx] = scaler_x.transform(X[:, scale_cols_idx])
    pred = model_predict(model, X)
    pred = inverse_transform_output(pred, scaler_y)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    w = np.asarray(weights, dtype=float).reshape(-1, 1)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    return (pred * w).sum(axis=0) / w_sum



def get_onehot_global_col_index(local_oh_index, oh_index_map):
    return oh_index_map[local_oh_index]

# ==============================================================
#            ① 复用原 2D 逻辑 → 包成函数
# ==============================================================

def heatmap_2d_inference(model, x_name, y_name,
                         stats_dict, x_col_names, scale_cols_idx,
                         scaler_x, scaler_y,
                         outdir_m, n_points=50,
                         group_templates=None, group_weights=None):
    """把你原先那段 2D 推断代码完整挪进来──除了 x_name/y_name 改成形参"""
    if (x_name not in stats_dict["continuous_cols"]
            or y_name not in stats_dict["continuous_cols"]):
        print(f"[WARN] {x_name}/{y_name} 不在连续列中 => 跳过 2D")
        return

    xinfo = stats_dict["continuous_cols"][x_name]
    yinfo = stats_dict["continuous_cols"][y_name]

    xv = np.linspace(xinfo["min"], xinfo["max"], n_points)
    yv = np.linspace(yinfo["min"], yinfo["max"], n_points)
    grid_x, grid_y = np.meshgrid(xv, yv)

    # —— baseline（训练集均值） ——
    base_vec = _get_base_vector(stats_dict, x_col_names)
    if group_templates is None or group_weights is None:
        group_templates = base_vec.reshape(1, -1)
        group_weights = np.ones(1, dtype=float)

    tmp = base_vec.reshape(1, -1)
    if scaler_x is not None:
        tmp[:, scale_cols_idx] = scaler_x.transform(tmp[:, scale_cols_idx])
    out_dim = model_predict(model, tmp).shape[-1]

    H, W = grid_x.shape
    heatmap_pred = np.zeros((H, W, out_dim))

    for i in trange(H, desc=f"2D({x_name},{y_name})", ncols=100):
        for j in range(W):
            batch = group_templates.copy()
            batch[:, x_col_names.index(x_name)] = grid_x[i, j]
            batch[:, x_col_names.index(y_name)] = grid_y[i, j]
            real = _weighted_predict(model, batch, group_weights, scaler_x, scaler_y, scale_cols_idx)
            heatmap_pred[i, j, :] = np.maximum(real.reshape(-1), 0)

    # —— 保存 ——
    tag = f"{x_name}__{y_name}".replace(" ", "_").replace("/", "_")
    np.save(os.path.join(outdir_m, f"grid_x_{tag}.npy"), grid_x)
    np.save(os.path.join(outdir_m, f"grid_y_{tag}.npy"), grid_y)
    np.save(os.path.join(outdir_m, f"heatmap_pred_{tag}.npy"), heatmap_pred)
    print(f"[INFO] 2D heatmap ({x_name},{y_name}) saved → {outdir_m}")


# ==============================================================
#            ② 三变量 3D 推断（透明等值面用）
# ==============================================================

def heatmap_3d_inference(model, axes_names, stats_dict,
                         x_col_names, scale_cols_idx,
                         scaler_x, scaler_y,
                         outdir_m, n_points=40,
                         group_templates=None, group_weights=None):
    """axes_names = [x_name, y_name, z_name]"""
    x_name, y_name, z_name = axes_names

    def _mm(col):
        info = stats_dict["continuous_cols"][col]
        return info["min"], info["max"]

    xv = np.linspace(*_mm(x_name), n_points)
    yv = np.linspace(*_mm(y_name), n_points)
    zv = np.linspace(*_mm(z_name), n_points)
    grid_x, grid_y, grid_z = np.meshgrid(xv, yv, zv, indexing="ij")

    base_vec = _get_base_vector(stats_dict, x_col_names)
    if group_templates is None or group_weights is None:
        group_templates = base_vec.reshape(1, -1)
        group_weights = np.ones(1, dtype=float)

    tmp = base_vec.reshape(1, -1)
    if scaler_x is not None:
        tmp[:, scale_cols_idx] = scaler_x.transform(tmp[:, scale_cols_idx])
    out_dim = model_predict(model, tmp).shape[-1]

    H, W, D = grid_x.shape
    heatmap_pred = np.zeros((H, W, D, out_dim))

    for i in trange(H, desc="3DHeatmap-X", ncols=100):
        for j in range(W):
            for k in range(D):
                batch = group_templates.copy()
                batch[:, x_col_names.index(x_name)] = grid_x[i, j, k]
                batch[:, x_col_names.index(y_name)] = grid_y[i, j, k]
                batch[:, x_col_names.index(z_name)] = grid_z[i, j, k]
                real = _weighted_predict(model, batch, group_weights, scaler_x, scaler_y, scale_cols_idx)
                heatmap_pred[i, j, k, :] = np.maximum(real.reshape(-1), 0)

    np.save(os.path.join(outdir_m, "grid_x_3d.npy"), grid_x)
    np.save(os.path.join(outdir_m, "grid_y_3d.npy"), grid_y)
    np.save(os.path.join(outdir_m, "grid_z_3d.npy"), grid_z)
    np.save(os.path.join(outdir_m, "heatmap_pred_3d.npy"), heatmap_pred)
    print(f"[INFO] 3D heatmap saved → {outdir_m}")

# --------------------------------------------------
#                    主入口
# --------------------------------------------------
def inference_main():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    if not os.path.isabs(csv_path):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(config_path), ".."))
        csv_path = os.path.join(repo_root, csv_path)
        config["data"]["path"] = csv_path

    inf_models = config["inference"].get("models", [])
    if not inf_models:
        print("[INFO] No inference models => exit.")
        return

    # >>> PATCH: 先解析数据集名称 & metadata 路径 -----------------------------
    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    run_id = get_run_id(config)
    if not run_id:
        run_id = _find_latest_run_id(csv_name)
        if run_id:
            print(f"[INFO] RUN_ID not set; using latest run_id => {run_id}")
        else:
            print("[ERROR] RUN_ID not set and no previous run found. Please train first or set RUN_ID.")
            return
    root_model_dir = get_root_model_dir(csv_name, run_id=run_id)
    meta_path = os.path.join(root_model_dir, "metadata.pkl")
    # <<< PATCH ----------------------------------------------------------------

    if not os.path.exists(meta_path):
        print(f"[ERROR] metadata => {meta_path} missing. Please retrain the model.")
        return

    base_stats_dict = joblib.load(meta_path)
    random_seed = config.get("data", {}).get("random_seed", 42)
    max_combos = config.get("inference", {}).get("max_combinations", None)

    base_inf = get_postprocess_dir(csv_name, run_id, "inference")
    ensure_dir(base_inf)


    # =================================================
    #               循环每个模型做推断
    # =================================================
    for mtype in inf_models:
        print(f"\n=== Inference => {mtype} ===")
        outdir_m = os.path.join(base_inf, mtype)
        ensure_dir(outdir_m)

        try:
            model, x_col_names, y_col_names = load_inference_model(mtype, config, run_id=run_id)
        except (FileNotFoundError, RuntimeError) as e:
            print(e)
            _write_inference_error(outdir_m, mtype, e)
            continue
        except Exception as e:
            print(f"[ERROR] Inference failed for {mtype}: {e}")
            _write_inference_error(outdir_m, mtype, e)
            continue

        # --- scaler & per-model metadata ---
        model_dir = get_model_dir(csv_name, mtype, run_id=run_id)
        stats_dict = base_stats_dict
        meta_m = os.path.join(model_dir, "metadata.pkl")
        if os.path.exists(meta_m):
            stats_dict = joblib.load(meta_m)

        meta_x_cols = stats_dict.get("x_col_names")
        if meta_x_cols is not None and len(meta_x_cols) != len(x_col_names):
            raise RuntimeError(
                f"Feature dimension mismatch between metadata and model for {mtype}: "
                f"metadata has {len(meta_x_cols)}, model has {len(x_col_names)}. "
                "Please retrain the model."
            )

        numeric_cols_idx = stats_dict["numeric_cols_idx"]
        scale_cols_idx_default = stats_dict.get("scale_cols_idx", numeric_cols_idx)
        scale_cols_idx_by_model = stats_dict.get("scale_cols_idx_by_model", {})
        onehot_groups = stats_dict.get("onehot_groups", [])
        group_names = stats_dict.get("group_names", [])
        group_value_vectors = stats_dict.get("group_value_vectors", {})

        sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        scaler_x = load_scaler(sx_path) if os.path.exists(sx_path) else None
        scaler_y = load_scaler(sy_path) if os.path.exists(sy_path) else None
        scale_cols_idx = scale_cols_idx_by_model.get(mtype, scale_cols_idx_default)
        scale_idx_path = os.path.join(model_dir, f"scale_cols_idx_{mtype}.npy")
        if os.path.exists(scale_idx_path):
            scale_cols_idx = np.load(scale_idx_path).tolist()

        # 校验 numeric 列一致性（仅当 scaler_x 存在）
        if scaler_x and len(scale_cols_idx) != scaler_x.n_features_in_:
            raise RuntimeError(
                f"scale_cols_idx ({len(scale_cols_idx)}) 与 "
                f"scaler_x.n_features_in_ ({scaler_x.n_features_in_}) 不匹配！"
            )
        # ----------------------------------------------------------
        # 根据 heatmap_axes 的长度自动分支
        # ----------------------------------------------------------
        axes_names = config["inference"].get("heatmap_axes", [])
        dim_axes = len(axes_names)
        n_points = config["inference"].get("n_points", 50)
        enable_3d = config["inference"].get("enable_3d_heatmap", True)
        skip_3d_models = set(config["inference"].get("skip_3d_models", []))

        base_vec = _get_base_vector(stats_dict, x_col_names)
        group_entries = _get_group_entries(stats_dict, x_col_names)
        group_templates, group_weights = _build_combo_templates(
            base_vec,
            group_entries,
            fixed=None,
            max_combos=max_combos,
            seed=random_seed
        )

        if dim_axes == 2:
            # 直接 2D
            heatmap_2d_inference(model,
                                 axes_names[0], axes_names[1],
                                 stats_dict, x_col_names, scale_cols_idx,
                                 scaler_x, scaler_y,
                                 outdir_m, n_points,
                                 group_templates=group_templates,
                                 group_weights=group_weights)

        elif dim_axes == 3:
            # ① 先跑 C(3,2) 三张 2D
            for x_name, y_name in combinations(axes_names, 2):
                heatmap_2d_inference(model,
                                     x_name, y_name,
                                     stats_dict, x_col_names, scale_cols_idx,
                                     scaler_x, scaler_y,
                                     outdir_m, n_points,
                                     group_templates=group_templates,
                                     group_weights=group_weights)
            # ② 再跑 3D
            if enable_3d and mtype not in skip_3d_models:
                heatmap_3d_inference(model,
                                     axes_names,
                                     stats_dict, x_col_names, scale_cols_idx,
                                     scaler_x, scaler_y,
                                     outdir_m, n_points,
                                     group_templates=group_templates,
                                     group_weights=group_weights)
            else:
                print(f"[INFO] Skip 3D heatmap for {mtype}.")
        else:
            print(f"[WARN] heatmap_axes={axes_names} (维数={dim_axes}) 非 2/3，已跳过连续变量可视化。")

        # =================================================
        #               B) Confusion‑like 输出
        # =================================================
        if len(onehot_groups) < 2:
            print("[WARN] Not enough groups => skip confusion.")
            continue

        conf_default = config["inference"]["confusion_axes"]
        conf_by_model = config["inference"].get("confusion_axes_by_model", {})
        conf_m = conf_by_model.get(mtype, {})
        row_kw = conf_m.get("row_name", conf_default["row_name"])
        col_kw = conf_m.get("col_name", conf_default["col_name"])

        row_idx = find_group_idx_by_name(row_kw, group_names) if group_names else None
        col_idx = find_group_idx_by_name(col_kw, group_names) if group_names else None

        if (row_idx is None) or (col_idx is None):
            row_idx = find_group_idx(row_kw, onehot_groups, x_col_names)
            col_idx = find_group_idx(col_kw, onehot_groups, x_col_names)

        if (row_idx is None) or (col_idx is None):
            print("[WARN] 指定的 row/col 关键字没找到 —— 退回前两组")
            row_idx, col_idx = 0, 1

        if row_idx == col_idx:
            print("[WARN] row_name 与 col_name 落在同一组 => skip confusion.")
            continue

        grpA = onehot_groups[row_idx]
        grpB = onehot_groups[col_idx]

        row_name = group_names[row_idx] if row_idx < len(group_names) else f"group_{row_idx}"
        col_name = group_names[col_idx] if col_idx < len(group_names) else f"group_{col_idx}"

        row_info = group_value_vectors.get(row_name)
        col_info = group_value_vectors.get(col_name)
        if not row_info or not col_info:
            print("[WARN] Missing group value vectors => skip confusion.")
            continue

        row_vals = row_info["values"]
        row_vecs = np.asarray(row_info["vectors"])
        col_vals = col_info["values"]
        col_vecs = np.asarray(col_info["vectors"])

        if row_vecs.shape[1] != len(grpA) or col_vecs.shape[1] != len(grpB):
            print("[WARN] Group vector dim mismatch => skip confusion.")
            continue

        if max_combos and (len(row_vals) * len(col_vals) > max_combos):
            cap = max(1, int(np.floor(np.sqrt(max_combos))))
            row_vals = row_vals[:cap]
            row_vecs = row_vecs[:cap]
            col_vals = col_vals[:cap]
            col_vecs = col_vecs[:cap]

        if "base_vec" not in locals():
            base_vec = _get_base_vector(stats_dict, x_col_names)
        if "group_entries" not in locals():
            group_entries = _get_group_entries(stats_dict, x_col_names)

        other_groups = [g for g in group_entries if g["gid"] not in {row_idx, col_idx}]
        other_templates, other_weights = _build_combo_templates(
            base_vec,
            other_groups,
            fixed=None,
            max_combos=max_combos,
            seed=random_seed
        )

        tmp = base_vec.reshape(1, -1)
        if scaler_x is not None:
            tmp[:, scale_cols_idx] = scaler_x.transform(tmp[:, scale_cols_idx])
        outdim = model_predict(model, tmp).shape[-1]

        confusion_pred = np.zeros((len(row_vals), len(col_vals), outdim), dtype=float)

        for i in trange(len(row_vals), desc="Confusion Rows", ncols=100):
            for j in range(len(col_vals)):
                batch = other_templates.copy()
                batch[:, grpA] = row_vecs[i]
                batch[:, grpB] = col_vecs[j]
                real_pred = _weighted_predict(
                    model, batch, other_weights, scaler_x, scaler_y, scale_cols_idx
                )
                confusion_pred[i, j, :] = real_pred.reshape(-1)

        np.save(os.path.join(outdir_m, "confusion_row_labels.npy"),
                np.array(row_vals, dtype=object))
        np.save(os.path.join(outdir_m, "confusion_col_labels.npy"),
                np.array(col_vals, dtype=object))
        # 循环结束之后（np.save 之前）插入
        v_min = confusion_pred.min()
        v_max = confusion_pred.max()
        eps = 1e-12  # 防除零
        confusion_norm = (confusion_pred - v_min) / (v_max - v_min + eps)

        # 新增一份归一化后的矩阵，名字示例：
        np.save(os.path.join(outdir_m, "confusion_pred_norm.npy"), confusion_norm)
        print(f"[INFO] confusion saved => {outdir_m}")


if __name__ == "__main__":
    inference_main()
