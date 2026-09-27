"""
training_script.py

BTC/ETH volatility + direction forecasting: training and evaluation entry
point. Loss functions, the training loop for `VolatilityModel`, the
XGBoost-based returns/direction model, validation/test evaluation, SHAP
feature attribution, and diagnostic plots.

Pulls data preparation from `data_preprocessing.py` and the model
architecture from `model_architecture.py`:

    from data_preprocessing import load_raw_data, build_alpha_feature_matrix, prepare_pytorch_data
    from model_architecture import VolatilityModel, VolWrapper

Requires (in addition to data_preprocessing.py's and model_architecture.py's
requirements):
    pip install shap xgboost tqdm
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import shap
import xgboost as xgb
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from data_preprocessing import load_raw_data, build_alpha_feature_matrix, prepare_pytorch_data
from model_architecture import VolatilityModel, VolWrapper


# ==========================================
# 1. LOSSES
# ==========================================

def exponential_asymmetric_loss(pred, target, penalty_multiplier=2.0):
    """
    Squared-error loss that penalizes UNDER-estimation of volatility more
    heavily than over-estimation, controlled by `penalty_multiplier`.
    Used to train `VolatilityModel`.
    """
    error = target - pred
    base_loss = torch.square(error)
    under_estimation = (error > 0).float()
    loss = base_loss * (1.0 + (under_estimation * penalty_multiplier))
    return torch.mean(loss)


def focal_loss_multiclass(pred, target, alpha=None, gamma=2.0):
    """
    Computes the Focal Loss for multi-class classification.

    Args:
        pred (torch.Tensor): Logits from the model (shape: [batch_size, num_classes])
        target (torch.Tensor): Ground truth labels (shape: [batch_size])
        alpha (list or torch.Tensor, optional): Class weights (e.g., [0.2, 0.5, 0.3])
        gamma (float): Focusing parameter. Default is 2.0.

    NOTE: kept for reference only. The returns model no longer uses this
    torch version -- it was replaced by a neural `ReturnModel` (since
    removed), which is itself now replaced by an XGBoost booster using the
    numpy-based `focal_loss_xgb_multiclass` objective below (same loss
    *intent*, just a custom grad/hess objective since XGBoost has no
    autograd).
    """
    ce_loss = F.cross_entropy(pred, target, reduction='none')
    probs = F.softmax(pred, dim=1)
    p_t = probs.gather(1, target.unsqueeze(1)).squeeze(1)
    modulating_factor = (1.0 - p_t) ** gamma
    loss = modulating_factor * ce_loss

    if alpha is not None:
        if isinstance(alpha, list):
            alpha = torch.tensor(alpha, device=pred.device)
        alpha_t = alpha.gather(0, target)
        loss = alpha_t * loss

    return torch.mean(loss)


def focal_loss_xgb_multiclass(preds, dtrain, alpha=None, gamma=2.0, num_class=3):
    """
    Custom multi-class focal-loss OBJECTIVE for XGBoost (returns model).

    XGBoost's `obj` callback must return (grad, hess) as flat arrays. There
    is no autograd here, so this uses the same diagonal approximation
    XGBoost's own built-in `multi:softmax`/`multi:softprob` objective uses
    for cross-entropy (grad = p - y, hess = 2*p*(1-p), per-class,
    ignoring cross-class Hessian terms) and scales both by the per-sample
    focal modulating factor `alpha_t * (1 - p_t) ** gamma` -- the same two
    ingredients (alpha-weighting + focusing term) as `focal_loss_multiclass`.

    Args:
        preds: raw margins from the booster, flat array of shape (n*num_class,)
        dtrain: xgb.DMatrix with integer class labels (0, 1, 2)
        alpha: list of per-class weights, e.g. [0.4, 0.2, 0.4]
        gamma: focusing parameter
        num_class: number of classes (3: Down/Chop/Up)
    """
    labels = dtrain.get_label().astype(int)
    n = labels.shape[0]
    preds = preds.reshape(n, num_class)

    z = preds - preds.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)

    y_onehot = np.zeros((n, num_class))
    y_onehot[np.arange(n), labels] = 1.0

    if alpha is None:
        alpha_arr = np.ones(n)
    else:
        alpha_vec = np.asarray(alpha)
        alpha_arr = alpha_vec[labels]

    p_t = np.clip(probs[np.arange(n), labels], 1e-7, 1 - 1e-7)
    focal_weight = alpha_arr * (1.0 - p_t) ** gamma

    grad = (probs - y_onehot) * focal_weight[:, None]
    hess = (2.0 * probs * (1.0 - probs)) * focal_weight[:, None]
    hess = np.clip(hess, 1e-6, None)  # XGBoost requires strictly positive hessian

    return grad.flatten(), hess.flatten()


def focal_eval_metric_xgb(preds, dtrain, alpha=None, gamma=2.0, num_class=3):
    """Custom eval metric matching `focal_loss_xgb_multiclass`, so training
    logs show the actual focal loss rather than XGBoost's default metric."""
    labels = dtrain.get_label().astype(int)
    n = labels.shape[0]
    preds = preds.reshape(n, num_class)

    z = preds - preds.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=1, keepdims=True)
    p_t = np.clip(probs[np.arange(n), labels], 1e-7, 1 - 1e-7)

    if alpha is None:
        alpha_arr = np.ones(n)
    else:
        alpha_vec = np.asarray(alpha)
        alpha_arr = alpha_vec[labels]

    loss = -alpha_arr * (1.0 - p_t) ** gamma * np.log(p_t)
    return 'focal_loss', float(np.mean(loss))


# ==========================================
# 2. EVALUATION HELPER
# ==========================================

def evaluate(df_raw, name, vol_model, ret_model, feature_cols, feature_cols_ret,
             feature_scaler_vol, seq_len, device, num_class=3, batch_size=256):
    """
    Runs both the volatility model (PyTorch, sequence-windowed) and the
    returns model (XGBoost, row-level tabular) on a raw split and reports
    MAE / directional accuracy.

    Returns
    -------
    pred_vol, y_vol_t, ret_probs, y_ret_flat
    """
    df_alpha, _, _ = build_alpha_feature_matrix(df_raw)

    # --- Volatility model: sequence-windowed, PyTorch ---
    X_vol, y_vol, _, _ = prepare_pytorch_data(
        df_alpha, feature_cols, seq_length=seq_len, fit_scaler=feature_scaler_vol
    )
    eval_dataset = TensorDataset(torch.tensor(X_vol).float(), torch.tensor(y_vol).float())
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    vol_model.eval()
    all_pred_vol, all_y_vol = [], []
    with torch.no_grad():
        for batch_x_vol, batch_y_vol in tqdm(eval_loader, desc=f"Evaluating ({name})", leave=False):
            batch_x_vol = batch_x_vol.to(device)
            pred_vol = vol_model(batch_x_vol)
            all_pred_vol.append(pred_vol.cpu())
            all_y_vol.append(batch_y_vol)

    pred_vol = torch.cat(all_pred_vol)
    y_vol_t = torch.cat(all_y_vol)
    mae_vol = torch.mean(torch.abs(pred_vol - y_vol_t)).item()
    print(f"{name} MAE (Volatility): {mae_vol:.4f}")

    # --- Returns model: XGBoost, row-level tabular features. No sequence
    # windowing, so these predictions are aligned one-to-one with
    # df_alpha's rows -- NOT candle-for-candle aligned with the
    # (seq_len-offset) volatility predictions above.
    X_ret_flat = df_alpha[feature_cols_ret].values.astype(np.float32)
    y_ret_flat = np.clip(np.round(df_alpha['target_ret'].values), 0, 2).astype(int)

    dm_ret = xgb.DMatrix(X_ret_flat)
    # `output_margin=True` is required since `objective` is explicitly
    # `multi:softprob` -- without it, XGBoost would apply its own softmax
    # and hand back probabilities directly, silently double-softmaxing when
    # we then apply our own softmax below.
    raw_margins = ret_model.predict(dm_ret, output_margin=True).reshape(-1, num_class)
    z = raw_margins - raw_margins.max(axis=1, keepdims=True)
    exp_z = np.exp(z)
    ret_probs = exp_z / exp_z.sum(axis=1, keepdims=True)

    preds_class = ret_probs.argmax(axis=1)
    acc_ret = float((preds_class == y_ret_flat).mean())
    print(f"{name} Directional Accuracy (3-Class): {acc_ret * 100:.2f}%")

    return pred_vol.numpy(), y_vol_t.numpy(), ret_probs, y_ret_flat


# ==========================================
# 3. MAIN TRAINING / EVALUATION ENTRY POINT
# ==========================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Booting Environment on {device} ---")
    if device.type == "cpu":
        print("WARNING: running on CPU. PatchTST+LoRA+LSTM training will be "
              "noticeably slower per batch, with no output between epochs by "
              "default -- watch the per-batch progress bar below to confirm "
              "it's making progress rather than stuck.")

    SEQ_LEN = 10

    # ---- Data ----
    df_train, df_val, df_test = load_raw_data()

    print("1. Engineering Alpha Matrix (train)...")
    df_train_alpha, feature_cols, feature_cols_ret = build_alpha_feature_matrix(df_train)

    print("\n--- Target Return Distribution (Counts, train split) ---")
    print(df_train_alpha['target_ret'].value_counts())
    print("\n--- Target Return Distribution (Percentages, train split) ---")
    print(df_train_alpha['target_ret'].value_counts(normalize=True) * 100)

    print("\n2. Generating Sequences and Scaling (train)...")
    # Volatility model: sequence-windowed, original feature set, unchanged.
    X_train_vol, y_vol_train, y_ret_train, feature_scaler_vol = prepare_pytorch_data(
        df_train_alpha, feature_cols, seq_length=SEQ_LEN, fit_scaler=None
    )

    # Returns model (XGBoost): row-level tabular features, no sequence
    # windowing needed for a tree model. `target_ret` in `df_train_alpha`
    # is already forward-shifted (built inside build_alpha_feature_matrix),
    # so each row's own features map directly to its own forward label.
    X_train_ret_flat = df_train_alpha[feature_cols_ret].values.astype(np.float32)
    y_train_ret_flat = np.clip(np.round(df_train_alpha['target_ret'].values), 0, 2).astype(int)

    # --- Safeguard --- ensures targets are exactly 0, 1, or 2, undoing any
    # accidental scaling.
    y_ret_train = np.round(y_ret_train)
    y_ret_train = np.clip(y_ret_train, 0, 2)

    X_train_vol_tensor = torch.tensor(X_train_vol).float().to(device)
    y_vol_train_tensor = torch.tensor(y_vol_train).float().to(device)

    train_dataset = TensorDataset(X_train_vol_tensor, y_vol_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    print(f"Vol dataset built: {len(X_train_vol)} sequences of shape {X_train_vol.shape[1:]}")
    print(f"Ret dataset built: {len(X_train_ret_flat)} rows of shape {X_train_ret_flat.shape[1:]} (tabular, for XGBoost)")

    # ---- VolatilityModel ----
    print("\n3. Initializing VolatilityModel (Hybrid LoRA-LSTM)...")
    num_feats_vol = len(feature_cols)
    vol_model = VolatilityModel(num_features=num_feats_vol, seq_length=SEQ_LEN, lstm_hidden=64).to(device)
    print("VolatilityModel LoRA trainable params:")
    vol_model.transformer_with_lora.print_trainable_parameters()

    LEARNING_RATE = 1e-3
    EPOCHS = 5
    vol_optimizer = optim.Adam(vol_model.parameters(), lr=LEARNING_RATE)

    # Class weights for the (XGBoost) focal loss: penalize missing class 0
    # (Down) and class 2 (Up) more, care less about class 1 (Chop).
    alpha_weights = [0.4, 0.2, 0.4]
    NUM_CLASS = 3

    print("\n4. Commencing VolatilityModel Training Loop...")
    print(f"   {len(train_loader)} batches/epoch x {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        vol_model.train()
        epoch_loss_vol = 0.0

        batch_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}", leave=True)
        for batch_x_vol, batch_y_vol in batch_bar:
            vol_optimizer.zero_grad()
            pred_vol = vol_model(batch_x_vol)
            loss_vol = exponential_asymmetric_loss(pred_vol, batch_y_vol, penalty_multiplier=0.5)
            loss_vol.backward()
            torch.nn.utils.clip_grad_norm_(vol_model.parameters(), max_norm=1.0)
            vol_optimizer.step()

            epoch_loss_vol += loss_vol.item()
            batch_bar.set_postfix(loss=f"{loss_vol.item():.4f}")

        avg_loss_vol = epoch_loss_vol / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS} | Vol Loss (Asymmetric): {avg_loss_vol:.4f}")

    # ---- ReturnModel (XGBoost) ----
    print("\n5. Training ReturnModel (XGBoost, Focal-Loss Objective)...")
    dtrain_ret = xgb.DMatrix(X_train_ret_flat, label=y_train_ret_flat)
    xgb_params = {
        # IMPORTANT: even though `obj=` below fully overrides the actual
        # gradient/Hessian computation (so this objective's built-in loss
        # is never used), XGBoost still needs a recognized multi-class
        # `objective` string to correctly configure `num_output_group`
        # (= num_class) internally. Without this key, XGBoost silently
        # defaults to `reg:squarederror` (single output per row), which
        # trains "successfully" but makes `Booster.predict()` return a
        # flat (n_rows,) array instead of (n_rows, num_class).
        'objective': 'multi:softprob',
        'num_class': NUM_CLASS,
        'eta': 0.1,
        'max_depth': 5,
        'tree_method': 'hist',
        'disable_default_eval_metric': 1,  # we supply our own focal-loss custom_metric
    }
    ret_model = xgb.train(
        xgb_params,
        dtrain_ret,
        num_boost_round=200,
        obj=lambda preds, dtr: focal_loss_xgb_multiclass(
            preds, dtr, alpha=alpha_weights, gamma=2.0, num_class=NUM_CLASS
        ),
        # NOTE: `feval` was removed in recent XGBoost (2.x) in favor of
        # `custom_metric` -- same callback signature (preds, dtrain) ->
        # (metric_name, value), just a renamed kwarg. If you're on an older
        # XGBoost (<2.0) that still expects `feval`, swap the keyword back.
        custom_metric=lambda preds, dtr: focal_eval_metric_xgb(
            preds, dtr, alpha=alpha_weights, gamma=2.0, num_class=NUM_CLASS
        ),
        evals=[(dtrain_ret, 'train')],
        verbose_eval=20,
    )

    print("\n--- Training complete. Evaluating on validation and test sets. ---")

    # ---- Validation / test evaluation ----
    vol_pred_plot, vol_true_plot, ret_probs_plot, ret_true_plot = evaluate(
        df_val, "Validation", vol_model, ret_model, feature_cols, feature_cols_ret,
        feature_scaler_vol, SEQ_LEN, device, num_class=NUM_CLASS,
    )
    evaluate(
        df_test, "Test", vol_model, ret_model, feature_cols, feature_cols_ret,
        feature_scaler_vol, SEQ_LEN, device, num_class=NUM_CLASS,
    )

    # ---- SHAP feature attribution ----
    print("\n5. Computing SHAP feature attributions...")

    rng = np.random.default_rng(42)

    # --- VolatilityModel: GradientExplainer ---
    background_idx = rng.choice(len(X_train_vol_tensor), size=min(100, len(X_train_vol_tensor)), replace=False)
    background_vol = X_train_vol_tensor[background_idx]

    df_val_alpha, _, _ = build_alpha_feature_matrix(df_val)
    X_val_vol, _, _, _ = prepare_pytorch_data(df_val_alpha, feature_cols, seq_length=SEQ_LEN, fit_scaler=feature_scaler_vol)
    X_val_vol_tensor = torch.tensor(X_val_vol).float().to(device)
    explain_idx = rng.choice(len(X_val_vol_tensor), size=min(200, len(X_val_vol_tensor)), replace=False)
    explain_sample_vol = X_val_vol_tensor[explain_idx]

    vol_model.eval()

    cudnn_was_enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    try:
        wrapped_vol = VolWrapper(vol_model).to(device).eval()
        explainer = shap.GradientExplainer(wrapped_vol, background_vol)
        sv_vol = explainer.shap_values(explain_sample_vol)
        if isinstance(sv_vol, list):
            sv_vol = sv_vol[0]
        sv_vol = np.array(sv_vol)
        if sv_vol.ndim == 4:
            sv_vol = sv_vol.squeeze(-1)
        vol_importance = np.abs(sv_vol).sum(axis=1).mean(axis=0)
    finally:
        torch.backends.cudnn.enabled = cudnn_was_enabled

    # --- ReturnModel (XGBoost): TreeExplainer, row-level tabular features ---
    X_val_ret_flat = df_val_alpha[feature_cols_ret].values.astype(np.float32)
    explain_ret_idx = rng.choice(len(X_val_ret_flat), size=min(200, len(X_val_ret_flat)), replace=False)
    explain_sample_ret_flat = X_val_ret_flat[explain_ret_idx]

    tree_explainer = shap.TreeExplainer(ret_model)
    sv_ret = tree_explainer.shap_values(explain_sample_ret_flat)
    if isinstance(sv_ret, list):
        sv_ret_class = sv_ret[2]  # explain the 'Strong Up' class (index 2)
    else:
        sv_ret = np.array(sv_ret)
        if sv_ret.ndim == 3:
            sv_ret_class = sv_ret[:, :, 2]
        else:
            sv_ret_class = sv_ret
    ret_importance = np.abs(sv_ret_class).mean(axis=0)

    shap_df_vol = pd.Series(vol_importance, index=feature_cols, name='vol').sort_values(ascending=False).to_frame()
    shap_df_ret = pd.Series(ret_importance, index=feature_cols_ret, name='ret').sort_values(ascending=False).to_frame()

    print("\nSHAP feature importance -- VolatilityModel (mean |SHAP|, summed over sequence window):")
    print(shap_df_vol)
    print("\nSHAP feature importance -- ReturnModel / XGBoost (mean |SHAP|, row-level):")
    print(shap_df_ret)

    fig, (shap_ax1, shap_ax2) = plt.subplots(
        1, 2, figsize=(16, max(6, max(len(feature_cols), len(feature_cols_ret)) * 0.35))
    )
    shap_df_vol.plot(kind='barh', ax=shap_ax1, legend=False, color='orange')
    shap_ax1.set_title('SHAP Feature Importance -- VolatilityModel')
    shap_ax1.set_xlabel('Mean |SHAP value|')
    shap_df_ret.plot(kind='barh', ax=shap_ax2, legend=False, color='green')
    shap_ax2.set_title("SHAP Feature Importance -- ReturnModel/XGBoost (Strong UP)")
    shap_ax2.set_xlabel('Mean |SHAP value|')
    plt.tight_layout()
    plt.show()

    # ---- Plots ----
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(vol_true_plot[:200], label='Actual Volatility', color='black', alpha=0.6)
    ax1.plot(vol_pred_plot[:200], label='Predicted Volatility', color='orange', linestyle='--')
    ax1.set_title('Asymmetric Volatility Prediction (First 200 Candles, Validation) -- VolatilityModel')
    ax1.legend()

    ax2.plot(ret_probs_plot[:200, 2], label='Prob(Strong UP)', color='green', alpha=0.8)
    ax2.plot(ret_probs_plot[:200, 0], label='Prob(Strong DOWN)', color='red', alpha=0.8)
    ax2.axhline(0.5, color='black', linestyle=':', label='50% Confidence')
    ax2.set_title('Directional Probability Head (Validation - 3 Class) -- ReturnModel')
    ax2.legend()

    plt.xlabel('Time Steps')
    plt.tight_layout()
    plt.show()

    # ---- Optional: save normalized OHLCV CSVs for inspection ----
    # (computed AFTER feature engineering, does not feed the model)
    columns_to_normalize = [
        'open', 'high', 'low', 'close', 'volume',
        'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume',
    ]
    export_scaler = MinMaxScaler()
    df_train_export = df_train.copy()
    df_val_export = df_val.copy()
    df_test_export = df_test.copy()

    export_scaler.fit(df_train_export[columns_to_normalize])
    df_train_export[columns_to_normalize] = export_scaler.transform(df_train_export[columns_to_normalize])
    df_val_export[columns_to_normalize] = export_scaler.transform(df_val_export[columns_to_normalize])
    df_test_export[columns_to_normalize] = export_scaler.transform(df_test_export[columns_to_normalize])

    df_train_export.to_csv('df_train_normalized.csv', index=False)
    df_val_export.to_csv('df_val_normalized.csv', index=False)
    df_test_export.to_csv('df_test_normalized.csv', index=False)
    print("Saved normalized CSVs (for reference only -- not used in training).")

    return {
        "vol_model": vol_model,
        "ret_model": ret_model,
        "feature_cols": feature_cols,
        "feature_cols_ret": feature_cols_ret,
        "feature_scaler_vol": feature_scaler_vol,
    }


if __name__ == "__main__":
    main()
