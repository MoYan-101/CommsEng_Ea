import numpy as np


def _maybe_ravel_single_output(y):
    arr = np.asarray(y)
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.ravel()
    return y


def train_sklearn_model(model,
                        X_train, Y_train,
                        X_val=None, Y_val=None,
                        enable_early_stop: bool = False,
                        es_rounds: int = 50):
    """
    Train a sklearn‑style model (RF / DT) 或自定义薄包装 (CatBoostRegression / XGBRegression).
    """

    base_name = model.__class__.__name__
    if hasattr(model, "model"):
        base_name = model.model.__class__.__name__

    # sklearn single-output estimators expect y shape (n_samples,).
    # Keep CatBoost(MultiRMSE) / MultiOutputRegressor targets as 2D.
    keep_2d_target = base_name.startswith("CatBoost") or base_name.startswith("MultiOutputRegressor")
    y_train_fit = Y_train if keep_2d_target else _maybe_ravel_single_output(Y_train)
    y_val_fit = Y_val if keep_2d_target else _maybe_ravel_single_output(Y_val)

    if enable_early_stop and X_val is not None:

        fit_kwargs = dict(
            eval_set=[(X_val, y_val_fit)],
            verbose=False
        )

        if base_name.startswith("CatBoost"):
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = es_rounds
        elif base_name.startswith("XGB"):
            # XGBoost 3.x 走 set_params(early_stopping_rounds)，fit 不接收该参数
            pass
        else:
            fit_kwargs["early_stopping_rounds"] = es_rounds

        model.fit(X_train, y_train_fit, **fit_kwargs)
    else:
        model.fit(X_train, y_train_fit)

    return model
