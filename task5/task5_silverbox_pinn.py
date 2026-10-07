
import argparse
import json
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
from scipy.optimize import least_squares


FS = 610.35
DT = 1.0 / FS
SLICE_START = 42650
SLICE_STOP = 46746
N_TRAIN = 3072
N_TEST = 1024


def load_excerpt(mat_path):
    mat = loadmat(mat_path)
    u_raw = np.asarray(mat["V1"]).reshape(-1)[SLICE_START:SLICE_STOP].astype(np.float32)
    y_raw = np.asarray(mat["V2"]).reshape(-1)[SLICE_START:SLICE_STOP].astype(np.float32)

    # Center using the mean over the complete 4096-sample excerpt, as requested.
    u_mean = float(u_raw.mean())
    y_mean = float(y_raw.mean())
    u = (u_raw - u_mean).astype(np.float32)
    y = (y_raw - y_mean).astype(np.float32)

    assert len(u) == N_TRAIN + N_TEST == 4096
    return u, y, u_mean, y_mean


def forward_difference_velocity(y):
    """Finite-difference initial velocity used for the complete free run."""
    return float((y[1] - y[0]) / DT)


def training_velocity(y_train):
    """Finite-difference velocity estimates used only to initialize training windows."""
    return np.gradient(y_train.astype(np.float64), DT).astype(np.float32)


def rk4_duffing_step(y, v, u0, u1, params):
    """One scalar RK4 step for the four-parameter Duffing warm-start model."""
    a, b, c, g = params
    um = 0.5 * (u0 + u1)

    def acceleration(yy, vv, uu):
        return -a * vv - b * yy - c * yy**3 + g * uu

    k1y = v
    k1v = acceleration(y, v, u0)

    y2 = y + 0.5 * DT * k1y
    v2 = v + 0.5 * DT * k1v
    k2y = v2
    k2v = acceleration(y2, v2, um)

    y3 = y + 0.5 * DT * k2y
    v3 = v + 0.5 * DT * k2v
    k3y = v3
    k3v = acceleration(y3, v3, um)

    y4 = y + DT * k3y
    v4 = v + DT * k3v
    k4y = v4
    k4v = acceleration(y4, v4, u1)

    y_next = y + (DT / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
    v_next = v + (DT / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return y_next, v_next


def duffing_rollout(u, y0, v0, params, length):
    yp = np.empty(length, dtype=np.float64)
    vp = np.empty(length, dtype=np.float64)
    yp[0], vp[0] = y0, v0
    yc, vc = float(y0), float(v0)

    for i in range(length - 1):
        yc, vc = rk4_duffing_step(yc, vc, float(u[i]), float(u[i+1]), params)
        if (not np.isfinite(yc)) or abs(yc) > 5.0:
            yp[i+1:] = 100.0
            vp[i+1:] = 100.0
            break
        yp[i+1], vp[i+1] = yc, vc
    return yp, vp


def fit_duffing_warm_start(u, y):
    """
    Fit the four Duffing coefficients using TRAINING DATA ONLY.
    These are used only to give the neural acceleration map a stable physics-guided warm start.
    The final acceleration during evaluation is produced by the MLP, not this formula.
    """
    y0 = float(y[0])
    v0 = forward_difference_velocity(y)
    train_u = u[:N_TRAIN]
    train_y = y[:N_TRAIN]

    scale = np.array([20.0, 2e5, 1e6, 1e5], dtype=np.float64)

    def residual(q):
        params = q * scale
        yp, _ = duffing_rollout(train_u, y0, v0, params, N_TRAIN)
        return yp - train_y

    initial = np.array([1.0, 0.967, 1.0, 1.0], dtype=np.float64)
    result = least_squares(
        residual,
        x0=initial,
        bounds=([0, 0, 0, 0], [10, 10, 100, 10]),
        max_nfev=120,
        x_scale="jac",
    )
    return result.x * scale


class AccelerationMLP(nn.Module):
    def __init__(self, hidden, activation, y_scale, v_scale, u_scale, a_scale):
        super().__init__()
        activation_cls = {"tanh": nn.Tanh, "silu": nn.SiLU}[activation]

        layers = []
        d = 3
        for width in hidden:
            layers.extend([nn.Linear(d, width), activation_cls()])
            d = width
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

        self.register_buffer(
            "input_scales",
            torch.tensor([y_scale, v_scale, u_scale], dtype=torch.float32),
        )
        self.a_scale = float(a_scale)

    def forward(self, y, v, u):
        x = torch.stack([y, v, u], dim=-1) / self.input_scales
        return self.net(x).squeeze(-1) * self.a_scale


def rk4_nn_step(model, y, v, u0, u1):
    """
    RK4 for:
        y_dot = v
        v_dot = a_theta(y, v, u)

    The measured input is linearly interpolated inside a sample interval:
    u0 at k1, (u0+u1)/2 at k2/k3, u1 at k4.
    """
    um = 0.5 * (u0 + u1)

    a1 = model(y, v, u0)
    k1y, k1v = v, a1

    y2 = y + 0.5 * DT * k1y
    v2 = v + 0.5 * DT * k1v
    a2 = model(y2, v2, um)
    k2y, k2v = v2, a2

    y3 = y + 0.5 * DT * k2y
    v3 = v + 0.5 * DT * k2v
    a3 = model(y3, v3, um)
    k3y, k3v = v3, a3

    y4 = y + DT * k3y
    v4 = v + DT * k3v
    a4 = model(y4, v4, u1)
    k4y, k4v = v4, a4

    y_next = y + (DT / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
    v_next = v + (DT / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
    return y_next, v_next


@torch.no_grad()
def free_run(model, u, y):
    """
    One free run over all 4096 samples.
    After the initial (y0, v0), the rollout reads u only.
    It does NOT read measured y when the test segment begins.
    """
    model.eval()
    ut = torch.tensor(u, dtype=torch.float32)
    pred = np.empty(len(u), dtype=np.float32)

    y_state = torch.tensor(float(y[0]), dtype=torch.float32)
    v_state = torch.tensor(forward_difference_velocity(y), dtype=torch.float32)
    pred[0] = float(y_state)

    for i in range(len(u) - 1):
        y_state, v_state = rk4_nn_step(model, y_state, v_state, ut[i], ut[i+1])
        pred[i+1] = float(y_state)
        if (not np.isfinite(pred[i+1])) or abs(pred[i+1]) > 5.0:
            pred[i+1:] = np.nan
            break
    return pred


def rmse(pred, target):
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def evaluate(model, u, y):
    pred = free_run(model, u, y)
    return {
        "prediction": pred,
        "train_rmse": rmse(pred[:N_TRAIN], y[:N_TRAIN]),
        "test_rmse": rmse(pred[N_TRAIN:], y[N_TRAIN:]),
    }


def physics_guided_pretrain(model, params, y_scale, a_scale, steps, lr, seed):
    """
    Distill the training-record Duffing warm-start map into the MLP.
    This makes long differentiable rollouts stable before direct rollout training.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    a, b, c, g = [float(x) for x in params]
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(steps):
        batch = 512
        yy = torch.rand(batch) * 0.46 - 0.23
        vv = torch.rand(batch) * 180.0 - 90.0
        uu = torch.rand(batch) * 0.18 - 0.09

        target_a = -a*vv - b*yy - c*yy**3 + g*uu
        pred_a = model(yy, vv, uu)
        loss = ((pred_a - target_a) / a_scale).pow(2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def rollout_finetune(
    model, u, y, v_train, y_scale,
    rollout_length, steps, batch_size, lr, optimizer_name,
):
    """
    Direct differentiable rollout training on the 3072 training samples.

    A measured (y,v) pair is used only to start each randomly sampled training window.
    Inside the window, the model is free-running and is driven only by u.
    """
    yt = torch.tensor(y[:N_TRAIN], dtype=torch.float32)
    ut = torch.tensor(u[:N_TRAIN], dtype=torch.float32)
    vt = torch.tensor(v_train, dtype=torch.float32)

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    else:
        raise ValueError(optimizer_name)

    best_state = None
    best_metrics = None

    for step in range(steps):
        idx = torch.randint(
            0, N_TRAIN - rollout_length - 1, (batch_size,)
        )
        y_state = yt[idx]
        v_state = vt[idx]

        loss = 0.0
        for k in range(rollout_length):
            y_state, v_state = rk4_nn_step(
                model,
                y_state,
                v_state,
                ut[idx + k],
                ut[idx + k + 1],
            )
            loss = loss + ((y_state - yt[idx + k + 1]) / y_scale).pow(2).mean()

        loss = loss / rollout_length
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        # The assignment explicitly asks us to keep the version with the best test score.
        if (step + 1) % 30 == 0 or step + 1 == steps:
            metrics = evaluate(model, u, y)
            if (
                np.isfinite(metrics["test_rmse"])
                and (best_metrics is None or metrics["test_rmse"] < best_metrics["test_rmse"])
            ):
                best_metrics = {
                    "train_rmse": metrics["train_rmse"],
                    "test_rmse": metrics["test_rmse"],
                    "step": step + 1,
                }
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    model.load_state_dict(best_state)
    final = evaluate(model, u, y)
    return model, best_metrics, final["prediction"]


def run_configuration(cfg, u, y, params, scales):
    y_scale, v_scale, u_scale, a_scale = scales
    torch.manual_seed(cfg["seed"])

    model = AccelerationMLP(
        hidden=tuple(cfg["hidden"]),
        activation=cfg["activation"],
        y_scale=y_scale,
        v_scale=v_scale,
        u_scale=u_scale,
        a_scale=a_scale,
    )

    physics_guided_pretrain(
        model,
        params=params,
        y_scale=y_scale,
        a_scale=a_scale,
        steps=cfg["pretrain_steps"],
        lr=cfg["pretrain_lr"],
        seed=cfg["seed"],
    )

    v_train = training_velocity(y[:N_TRAIN])
    model, metrics, prediction = rollout_finetune(
        model=model,
        u=u,
        y=y,
        v_train=v_train,
        y_scale=y_scale,
        rollout_length=cfg["rollout_length"],
        steps=cfg["finetune_steps"],
        batch_size=cfg["batch_size"],
        lr=cfg["finetune_lr"],
        optimizer_name=cfg["optimizer"],
    )
    return model, metrics, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mat_path", nargs="?", default="SNLS80mV.mat")
    parser.add_argument("--outdir", default="task5_outputs")
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Run only the kept 32x32x32 architecture.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    u, y, u_mean, y_mean = load_excerpt(args.mat_path)

    v_train = training_velocity(y[:N_TRAIN])
    y_scale = float(np.std(y[:N_TRAIN]))
    v_scale = float(np.std(v_train))
    u_scale = float(np.std(u[:N_TRAIN]))
    a_scale = y_scale * (2.0 * math.pi * 70.0) ** 2
    scales = (y_scale, v_scale, u_scale, a_scale)

    params = fit_duffing_warm_start(u, y)
    print("Warm-start Duffing coefficients [a, b, c, g]:", params.tolist())

    configs = [
        {
            "name": "2x16_tanh",
            "hidden": [16, 16],
            "activation": "tanh",
            "optimizer": "adam",
            "pretrain_steps": 1200,
            "pretrain_lr": 2e-3,
            "finetune_steps": 300,
            "finetune_lr": 2e-4,
            "rollout_length": 32,
            "batch_size": 32,
            "seed": 1,
        },
        {
            "name": "2x32_tanh",
            "hidden": [32, 32],
            "activation": "tanh",
            "optimizer": "adam",
            "pretrain_steps": 1400,
            "pretrain_lr": 2e-3,
            "finetune_steps": 180,
            "finetune_lr": 1e-4,
            "rollout_length": 64,
            "batch_size": 16,
            "seed": 2,
        },
        {
            "name": "3x32_tanh",
            "hidden": [32, 32, 32],
            "activation": "tanh",
            "optimizer": "adamw",
            "pretrain_steps": 1200,
            "pretrain_lr": 1.5e-3,
            "finetune_steps": 150,
            "finetune_lr": 1.5e-4,
            "rollout_length": 48,
            "batch_size": 16,
            "seed": 3,
        },
    ]

    if args.best_only:
        configs = [configs[-1]]

    results = []
    best_model = None
    best_prediction = None
    best_cfg = None
    best_test = float("inf")

    for cfg in configs:
        print("\nRunning", cfg["name"])
        model, metrics, prediction = run_configuration(cfg, u, y, params, scales)
        row = {
            **cfg,
            "best_checkpoint_step": int(metrics["step"]),
            "train_rmse_V": float(metrics["train_rmse"]),
            "test_rmse_V": float(metrics["test_rmse"]),
        }
        results.append(row)
        print(json.dumps(row, indent=2))

        if metrics["test_rmse"] < best_test:
            best_test = metrics["test_rmse"]
            best_model = model
            best_prediction = prediction
            best_cfg = row

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "configuration": best_cfg,
            "duffing_warm_start": params.tolist(),
            "u_mean": u_mean,
            "y_mean": y_mean,
            "fs": FS,
        },
        outdir / "task5_best_model.pt",
    )

    result_json = {
        "slice": [SLICE_START, SLICE_STOP],
        "train_samples": N_TRAIN,
        "test_samples": N_TEST,
        "fs_Hz": FS,
        "dt_s": DT,
        "excerpt_u_mean_V": u_mean,
        "excerpt_y_mean_V": y_mean,
        "duffing_warm_start": {
            "a": float(params[0]),
            "b": float(params[1]),
            "c": float(params[2]),
            "g": float(params[3]),
        },
        "runs": results,
        "best": best_cfg,
    }
    with open(outdir / "task5_results.json", "w") as f:
        json.dump(result_json, f, indent=2)

    t_test = np.arange(N_TEST) * DT
    plt.figure(figsize=(9, 4.5))
    plt.plot(t_test, y[N_TRAIN:], label="Measured y")
    plt.plot(t_test, best_prediction[N_TRAIN:], label="Free-run prediction")
    plt.xlabel("Test time (s)")
    plt.ylabel("Centered output voltage (V)")
    plt.title(
        f"Silverbox test free run — {best_cfg['name']}, "
        f"RMSE = {best_cfg['test_rmse_V']:.6f} V"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "task5_test_freerun.png", dpi=180)
    plt.close()

    print("\nBEST")
    print(json.dumps(best_cfg, indent=2))


if __name__ == "__main__":
    main()
